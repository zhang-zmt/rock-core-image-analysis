"""只读评估已有岩性分类权重，不更新参数、不覆盖训练结果。

使用当前训练脚本的默认确定性预处理：短边255、中心裁剪224、ImageNet归一化。
权重未保存训练配置，相关默认设置仅是本次复算条件，不冒充历史训练日志。
"""
import argparse
import collections
import hashlib
import io
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def fingerprint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("读取期间文件变化：" + str(path))
    return {"sha256": digest.hexdigest(), "size": after.st_size, "mtime_ns": after.st_mtime_ns}


class EvaluationDataset(Dataset):
    """读取真实图片时一并核对SHA，避免把损坏或变动的样本静默跳过。"""
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["training_path"])
        before = path.stat()
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["crop_sha256"]:
            raise ValueError("裁片内容不匹配：" + str(path))
        with Image.open(io.BytesIO(data)) as image:
            if list(image.size) != row["size"] or getattr(image, "n_frames", 1) != 1:
                raise ValueError("图像尺寸或帧数与记录不符：" + str(path))
            tensor = self.transform(image.convert("RGB"))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("读取期间图片变化：" + str(path))
        return tensor, row["class_index"], index


def metrics(truth, probability, classes):
    """同时报告总体与逐类指标；没有分母时记为0并保留真实支持数。"""
    predicted = probability.argmax(axis=1)
    count = len(classes)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    precision = np.divide(true_positive, predicted_count, out=np.zeros(count), where=predicted_count != 0)
    recall = np.divide(true_positive, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count), where=(precision + recall) != 0)
    per_class = []
    for index, label in enumerate(classes):
        per_class.append({"class_index": index, "class_name": label,
                          "support": int(support[index]), "correct": int(true_positive[index]),
                          "predicted_count": int(predicted_count[index]),
                          "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])})
    top3 = np.argsort(-probability, axis=1)[:, :min(3, count)]
    confidence = probability.max(axis=1)
    wrong = predicted != truth
    return {"samples": len(truth), "correct": int((~wrong).sum()), "errors": int(wrong.sum()),
            "accuracy": float((~wrong).mean()), "top3_accuracy": float((top3 == truth[:, None]).any(axis=1).mean()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support) / support.sum()),
            "high_confidence_errors_ge_090": int((wrong & (confidence >= .9)).sum()),
            "confusion_matrix": confusion.tolist(), "per_class": per_class}


def evaluate_split(model, rows, classes, chinese, device, output, split, transform, batch_size, train_groups):
    loader = DataLoader(EvaluationDataset(rows, transform), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    probabilities = np.zeros((len(rows), len(classes)), dtype=np.float32)
    truth = np.zeros(len(rows), dtype=np.int64)
    loss_sum = 0.0
    forward_seconds = 0.0
    start = time.perf_counter()
    last_update = start
    processed = 0
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - tick
            if not torch.isfinite(logits).all():
                raise ValueError("模型输出存在非有限值")
            loss_sum += float(functional.cross_entropy(logits, labels, reduction="sum"))
            probabilities[indices.numpy()] = logits.softmax(dim=1).cpu().numpy()
            truth[indices.numpy()] = labels.cpu().numpy()
            processed += len(labels)
            now = time.perf_counter()
            if now - last_update >= 20 or processed == len(rows):
                print(f"{split}：{processed}/{len(rows)}，已用{now-start:.1f}秒", flush=True)
                last_update = now
    seconds = time.perf_counter() - start
    result = metrics(truth, probabilities, classes)
    result.update(cross_entropy=loss_sum / len(rows), evaluation_seconds=seconds,
                  forward_seconds=forward_seconds, images_per_second=len(rows) / seconds,
                  forward_ms_per_image=forward_seconds * 1000 / len(rows))
    group_indices = collections.defaultdict(list)
    for index, row in enumerate(rows):
        group_indices[row["group_id"]].append(index)
    group_truth = []
    group_probability = []
    group_records = []
    for group, indices in sorted(group_indices.items()):
        labels = set(truth[indices].tolist())
        if len(labels) != 1:
            raise ValueError("同一原图组出现多个目录标签，不能直接汇总")
        actual = next(iter(labels))
        mean_probability = probabilities[indices].mean(axis=0)
        group_truth.append(actual)
        group_probability.append(mean_probability)
        group_records.append({"group_id": group, "patch_count": len(indices),
                              "true_class": classes[actual], "predicted_class": classes[int(mean_probability.argmax())],
                              "also_in_train": group in train_groups,
                              "original_path": rows[indices[0]].get("original_path_metadata_only")})
    result["original_groups"] = len(group_indices)
    result["groups_overlapping_train"] = len(set(group_indices) & train_groups)
    result["independent_acquisition_groups_confirmed"] = None
    result["group_aggregation"] = metrics(np.array(group_truth), np.array(group_probability), classes)
    result["group_aggregation"]["method"] = "仅对本集合中同一原图的裁片平均概率；不是未见原图评估"
    for entry in result["per_class"]:
        entry["original_name"] = chinese[entry["class_name"]]
        entry["original_groups"] = len(set(rows[i]["group_id"] for i in np.flatnonzero(truth == entry["class_index"])))
    phase = output / split
    phase.mkdir()
    write_json(phase / "metrics.json", result)
    write_json(phase / "group_predictions.json", group_records)
    np.savez_compressed(phase / "probabilities.npz", truth=truth, probabilities=probabilities,
                        case_ids=np.array([r["case_id"] for r in rows]))
    error_count = 0
    with (phase / "predictions.jsonl").open("w", encoding="utf-8") as predictions, (phase / "errors.jsonl").open("w", encoding="utf-8") as errors:
        for index, row in enumerate(rows):
            top = np.argsort(-probabilities[index])[:3]
            record = {"case_id": row["case_id"], "split": split, "evaluation_round": "20261108_01",
                      "true_class": row["category"], "true_name": chinese[row["category"]],
                      "predicted_class": classes[int(top[0])], "predicted_name": chinese[classes[int(top[0])]],
                      "top3": [{"class_name": classes[int(k)], "original_name": chinese[classes[int(k)]],
                                "score": float(probabilities[index, k])} for k in top],
                      "image_path": row["training_path"], "group_id": row["group_id"],
                      "original_path": row.get("original_path_metadata_only"), "bbox_on_source": row.get("bbox_on_source"),
                      "quality_decision_source": row.get("decision_source"),
                      "quality_human_confirmed": row.get("human_confirmed"), "crop_sha256": row["crop_sha256"]}
            line = json.dumps(record, ensure_ascii=False) + "\n"
            predictions.write(line)
            if top[0] != truth[index]:
                errors.write(line)
                error_count += 1
    if error_count != result["errors"]:
        raise ValueError("错误清单与指标不一致")
    print(f"{split}结果：准确率{result['accuracy']:.6f}，Macro-F1 {result['macro_f1']:.6f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    root = args.project.resolve()
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    if output.exists():
        raise FileExistsError("拒绝覆盖评估输出")
    # 用一个可手算的小例子检查指标实现。
    check = metrics(np.array([0, 0, 1, 1]), np.array([[.8, .2], [.2, .8], [.2, .8], [.2, .8]]), ["a", "b"])
    if abs(check["accuracy"] - .75) > 1e-10 or abs(check["macro_f1"] - (2 / 3 + .8) / 2) > 1e-10:
        raise ValueError("指标自检失败")
    files = [checkpoint, root / "utils" / "train_best.py", root / "data_split" / "samples.jsonl",
             root / "data_split" / "class_to_idx.json", root / "training_metadata" / "class_labels.json"]
    for split in ("train", "val", "test"):
        files.append(root / "data_split" / (split + ".txt"))
    snapshots = {str(path): fingerprint(path) for path in files}
    mapping = json.loads((root / "data_split" / "class_to_idx.json").read_text(encoding="utf-8"))
    classes = sorted(mapping, key=mapping.get)
    if classes != sorted(classes) or [mapping[c] for c in classes] != list(range(len(classes))):
        raise ValueError("类别编号与训练代码读取顺序不一致")
    chinese = {r["class_name"]: r["original_name"] for r in json.loads((root / "training_metadata" / "class_labels.json").read_text(encoding="utf-8"))}
    lookup = {}
    for line in (root / "data_split" / "samples.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        key = os.path.normcase(str(Path(row["training_path"]).resolve()))
        if key in lookup:
            raise ValueError("溯源清单路径重复")
        lookup[key] = row
    partitions = {}
    seen = set()
    for split in ("train", "val", "test"):
        partitions[split] = []
        for line in (root / "data_split" / (split + ".txt")).read_text(encoding="utf-8").splitlines():
            path_text, category = line.split("\t")
            path = Path(path_text).resolve()
            key = os.path.normcase(str(path))
            row = lookup[key]
            if key in seen or row["split"] != split or row["category"] != category or not path.is_file():
                raise ValueError("训练清单、溯源或实际文件不一致")
            if root not in path.parents or row["class_index"] != mapping[category]:
                raise ValueError("路径或类别索引不正确")
            seen.add(key)
            partitions[split].append(row)
    if seen != set(lookup):
        raise ValueError("清单未完整覆盖当前数据")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    blocks = []
    for layer in (1, 2, 3, 4):
        indices = set()
        for key in state:
            if key.startswith(f"layer{layer}."):
                indices.add(int(key.split(".")[1]))
        blocks.append(len(indices))
    if blocks != [3, 4, 23, 3] or list(state["fc.weight"].shape) != [len(classes), 2048]:
        raise ValueError("权重结构不是预期的21类ResNet101")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = models.resnet101(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros(args.batch_size, 3, 224, 224, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / "evaluate_checkpoint.py")
    report = {"status": "running", "at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "architecture_verified": "resnet101", "class_count": len(classes), "checkpoint": str(checkpoint),
              "checkpoint_contains": "仅state_dict；无训练日志、类别名称、轮数、优化器和调度器状态",
              "historical_training_config_verified": False, "class_order_assumption": "沿用当前训练清单的排序和编号",
              "device": str(device), "device_name": "CPU", "batch_size": args.batch_size, "workers": 2,
              "torch": torch.__version__, "torchvision": torchvision.__version__, "python": sys.version,
              "preprocessing": "RGB; Resize短边255; CenterCrop224; ImageNet归一化",
              "input_shape": [3, 224, 224], "tta": False, "amp": False, "tf32": False,
              "parameter_count": sum(p.numel() for p in model.parameters()), "snapshots": snapshots,
              "measurement_scope": "evaluation_seconds含读图、SHA、预处理、传输、前向和结果汇总，不含模型加载、预热及结果文件写入；forward_seconds仅前向，GPU同步计时",
              "limitations": ["当前为裁片级划分，同源原图跨集合；不代表独立原图泛化", "更高层独立采集组未确认", "目录岩性标签没有在本次重新地质鉴定", "没有逐轮日志，无法恢复训练曲线或确认训练轮次", "训练集复算使用确定性预处理，与历史训练增强不同"],
              "results": {}, "errors": []}
    if device.type == "cuda":
        report["device_name"] = torch.cuda.get_device_name(0)
    write_json(output / "evaluation_summary.json", report)
    try:
        train_groups = set(row["group_id"] for row in partitions["train"])
        for split in ("test", "val", "train"):
            result = evaluate_split(model, partitions[split], classes, chinese, device, output, split, transform, args.batch_size, train_groups)
            report["results"][split] = result
            write_json(output / "evaluation_summary.json", report)
        for path, before in snapshots.items():
            if fingerprint(Path(path)) != before:
                raise ValueError("评估期间权重、代码或依据清单发生变化：" + path)
        report.update(status="completed", input_files_unchanged=True, decoded_and_sha_checked_images=len(seen))
        if device.type == "cuda":
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(output / "evaluation_summary.json", report)
        print("评估完成，权重和输入清单未修改。", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed_incomplete", errors=[repr(error)])
        write_json(output / "evaluation_summary.json", report)
        raise


if __name__ == "__main__":
    main()
