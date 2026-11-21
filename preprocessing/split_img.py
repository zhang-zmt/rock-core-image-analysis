#!/usr/bin/env python3
"""只读500×700初裁图，按原目录另存四张250×350无损PNG。

依赖Python>=3.10、Pillow。拒绝已有输出目录；不缩放，不画线，不做质量筛除。
--manifest可继承core_crop.py的原图来源、坐标、分组和补窗信息。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

from PIL import Image, ImageDraw

VERSION = '1.0.0'
# 支持的图片扩展名；大小写不敏感，其他扩展名在目录扫描时跳过、单文件输入时报错。
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# 允许直接处理的像素模式；其他模式（如 CMYK、浮点）报错而不自动转换位深。
PNG_MODES = {'1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I;16'}
# Windows 下禁止写入的源数据目录，防止污染原图或上一轮裁切结果。
PROTECTED_ROOTS = tuple(Path(p).resolve() for p in os.environ.get("ROCK_CORE_PROTECTED_ROOTS", "").split(os.pathsep) if p)
# 四象限在 500×700 父初裁图中的 xyxy 裁框：无重叠、无间隙，恰好拼回整图。
QUADRANTS = {
    'top_left': (0, 0, 250, 350), 'top_right': (250, 0, 500, 350),
    'bot_left': (0, 350, 250, 700), 'bot_right': (250, 350, 500, 700),
}


def sha256_file(path: Path) -> str:
    """按 1 MiB 分块读取计算文件 SHA-256，避免大文件一次性占用内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict:
    """取文件大小与修改时间（纳秒），用于处理前后的一致性比对。"""
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def inside(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 目录之内（先解析真实路径再做前缀比较）。"""
    return path.resolve().is_relative_to(root.resolve())


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """输入输出路径合法性检查：输入必须存在，输出必须尚不存在，且不嵌套、不写入受保护目录。"""
    source, output = source.resolve(strict=True), output.resolve()
    if not source.is_file() and not source.is_dir():
        raise ValueError('输入必须是图片文件或目录。')
    if output.exists():
        raise ValueError(f'输出目录已存在，禁止覆盖，请指定新目录：{output}')
    if source.is_dir() and (inside(output, source) or inside(source, output)):
        raise ValueError('输入和输出目录不能相同，也不能互相嵌套。')
    if any(inside(output, root) for root in PROTECTED_ROOTS):
        raise ValueError('禁止写入受保护的目录，请指定新的输出目录。')
    return source, output


def discover_images(source: Path) -> list[Path]:
    """递归收集图片：单文件直接返回；目录则按相对路径（casefold）排序，保证 --limit 结果稳定。"""
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'不支持的图片扩展名：{source.suffix}')
        return [source]
    found = []

    def walk_error(error):
        raise error  # 遍历中任何读取错误都中止，不静默跳过。

    # followlinks=False 且剔除符号链接/目录联接，避免重复或越出源目录地遍历。
    for directory, subdirs, files in os.walk(source, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        subdirs[:] = [name for name in subdirs if not (parent / name).is_symlink()
                      and not getattr(parent / name, 'is_junction', lambda: False)()]
        found.extend(parent / name for name in files
                     if (parent / name).suffix.lower() in EXTENSIONS and not (parent / name).is_symlink())
    return sorted(found, key=lambda path: path.relative_to(source).as_posix().casefold())


def _bbox(value, name: str) -> list[int]:
    """校验 manifest 中的裁框：4 个非负整数且恰好对应 500×700 的原图区域。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError(f'manifest的{name}必须是4个整数。')
    x0, y0, x1, y1 = value
    if x0 < 0 or y0 < 0 or (x1-x0, y1-y0) != (500, 700):
        raise ValueError(f'manifest的{name}必须对应500×700原图区域。')
    return list(value)


def load_parent_manifest(path: Path | None) -> dict[str, dict]:
    """读取上一轮 core_crop 的 manifest.jsonl，仅保留 stage=initial 记录。

    返回字典以初裁文件的绝对路径（normcase 后）为键。逐行严格校验：
    任何一行不合法都整体报错，避免带着错误的来源映射继续切分。
    """
    if path is None:
        return {}
    records = {}
    with path.open('r', encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('记录必须是对象。')
                if row.get('stage') != 'initial':
                    continue
                rel = row.get('output_relpath')
                if not isinstance(rel, str) or not rel or '\\' in rel or ':' in rel:
                    raise ValueError('output_relpath必须为相对路径，使用/分隔。')
                pure = PurePosixPath(rel)
                if pure.is_absolute() or '..' in pure.parts:
                    raise ValueError('output_relpath不能是绝对路径或包含..。')
                if row.get('size') != [500, 700]:
                    raise ValueError('初裁尺寸必须为[500,700]。')
                _bbox(row.get('bbox_on_source'), 'bbox_on_source')
                if not isinstance(row.get('source_id'), str) or not row['source_id']:
                    raise ValueError('缺少原图source_id，不能建立原图分组。')
                for field, bound in [('overlap_x_px', 500), ('overlap_y_px', 700)]:
                    v = row.get(field, 0)
                    if type(v) is not int or not 0 <= v <= bound:
                        raise ValueError(f'{field}不合法。')
                # output_relpath 以父清单所在目录为基准，解析出初裁文件的实际位置。
                resolved = (path.parent / Path(*pure.parts)).resolve()
                if not inside(resolved, path.parent):
                    raise ValueError('清单输出路径必须位于清单所在目录内。')
                key = os.path.normcase(str(resolved))
                if key in records:
                    raise ValueError('相同初裁路径有重复记录。')
                records[key] = row
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'manifest第{line_number}行：{exc}') from exc
    return records


def save_verified_png(image: Image.Image, target: Path) -> None:
    """写出 PNG 并重新解码逐像素核对；任何不一致（含调色板）都视为写出失败。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        image.save(stream, format='PNG', compress_level=3)
    # 每一张正式输出均重新解码核对，不只检查PNG标称尺寸。
    with Image.open(target) as saved:
        saved.load()
        if saved.size != image.size or saved.mode != image.mode or saved.tobytes() != image.tobytes():
            raise RuntimeError(f'写出后的像素核验失败：{target}')
        if image.mode == 'P' and (saved.getpalette() != image.getpalette()
                                  or saved.info.get('transparency') != image.info.get('transparency')):
            raise RuntimeError(f'写出后的调色板核验失败：{target}')


def save_preview(image: Image.Image, target: Path) -> None:
    """生成独立预览：左侧 500×700 父图叠加红色十字切分线，右侧为四象限缩略图。"""
    # 预览展示原始尺寸的父图和四象限；红线只画在副本上。
    with Image.new('RGB', (1160, 850), '#f4f6f8') as canvas:
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), 'Parent 500x700; red cross = split lines (preview only)', fill='black')
        canvas.paste(image.convert('RGB'), (20, 65))
        draw.line((270, 65, 270, 764), fill='#ff2222', width=2)
        draw.line((20, 415, 519, 415), fill='#ff2222', width=2)
        for i, (name, box) in enumerate(QUADRANTS.items()):
            x, y = 610 + (i % 2) * 270, 65 + (i // 2) * 390
            draw.text((x, y-22), f'{name} 250x350', fill='black')
            with image.crop(box) as patch:
                canvas.paste(patch.convert('RGB'), (x, y))
        draw.text((20, 800), 'No resize, interpolation, padding or quality filtering. PNG output preserves pixels.', fill='black')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            canvas.save(stream, format='PNG')


def quadrant_record(parent: dict | None, path: Path, relative: Path, digest: str,
                    name: str, qbox: tuple[int, int, int, int], qrel: Path) -> dict:
    """构造一张四象限输出的 manifest 记录（schema 版本 3）。

    parent 为父初裁的 manifest 记录；没有父清单时生成 provenance_status=
    parent_only 的记录，原图来源、原图坐标与 group_id 置空，不冒充分组。
    """
    row = dict(parent or {})
    x0, y0, x1, y1 = qbox
    original_bbox = parent['bbox_on_source'] if parent else None
    # 父窗与前窗的交叠带位于父窗左/上边缘内侧，取象限与该带的交集长度，
    # 不直接照抄父窗整段重叠值。
    ox = max(0, min(x1, parent.get('overlap_x_px', 0)) - x0) if parent else None
    oy = max(0, min(y1, parent.get('overlap_y_px', 0)) - y0) if parent else None
    row.update({
        'manifest_schema_version': 3, 'splitter_version': VERSION, 'stage': 'quadrant',
        'quadrant': name, 'size': [250, 350], 'output_relpath': qrel.as_posix(),
        'parent_path': str(path), 'parent_relpath': relative.as_posix(),
        'parent_relpath_base': 'input_root', 'parent_sha256': digest,
        'parent_manifest_output_relpath': parent['output_relpath'] if parent else None,
        'parent_size': [500, 700], 'bbox_on_initial': list(qbox),
        'parent_bbox_on_source': original_bbox,
        'bbox_on_source': [original_bbox[0]+x0, original_bbox[1]+y0,
                           original_bbox[0]+x1, original_bbox[1]+y1] if parent else None,
        'provenance_status': 'original_source_known' if parent else 'parent_only',
        'group_id': parent['source_id'] if parent else None,
        # 补尾标记沿用父初裁窗口；父级重叠值原样保留，供追溯对照。
        'tail_flags_scope': 'parent_initial_window' if parent else None,
        'parent_overlap_x_px': parent.get('overlap_x_px', 0) if parent else None,
        'parent_overlap_y_px': parent.get('overlap_y_px', 0) if parent else None,
        'parent_overlap_area_px': parent.get('overlap_area_px', 0) if parent else None,
        'overlap_x_px': ox, 'overlap_y_px': oy,
        # 象限自身的重叠面积按横纵两条交叠带的并集计算（250×350 尺度）。
        'overlap_area_px': ox*350 + oy*250 - ox*oy if parent else None,
        'overlap_scope': 'earlier_quadrants_of_previous_initial_windows' if parent else None,
    })
    if parent is None:
        row.update(source_path=None, source_relpath=None, source_sha256=None, source_id=None)
    return row


def check_space(output: Path, count: int, preview: bool) -> int:
    """按每张父图 4 张 250×350 输出、每像素 4 字节保守估算所需磁盘空间（字节）。"""
    required = int(count * 500 * 700 * 4 * 1.02) + 64*1024*1024 + (1160*850*4 if preview else 0)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required:
        raise ValueError(f'可用空间不足，保守估算需要{required/1024**3:.2f} GiB。')
    return required


def run(args: argparse.Namespace) -> dict:
    """主流程：路径校验→输入预检→（dry-run 返回计划）→建目录→逐图切分→完整性核对→写 run_summary。

    任一图片在正式处理阶段失败即中止后续处理，保留已写结果与错误记录；
    返回的 summary 由 main 换算成退出码。
    """
    source, output = validate_paths(args.input, args.output)
    if args.limit < 0:
        raise ValueError('--limit不能小于0。')
    source_root = source if source.is_dir() else source.parent
    all_images = discover_images(source)
    # 预检前的全量文件快照，结束后用于核对整个输入树是否发生过变化。
    inventory = {p.relative_to(source_root).as_posix(): file_state(p) for p in all_images}
    images = all_images[:args.limit] if args.limit else all_images
    if not images:
        raise ValueError('未找到支持的图片。')
    manifest_path = args.manifest.resolve(strict=True) if args.manifest else None
    # 记录父清单处理前的大小/修改时间/摘要，结束后复查其未被改动。
    reference_before = ({'path': str(manifest_path), **file_state(manifest_path),
                         'sha256': sha256_file(manifest_path)} if manifest_path else None)
    parent_records = load_parent_manifest(manifest_path)
    jobs, targets = [], set()
    # 全部所选初裁图先预检；尺寸、帧数、模式、EXIF 方向与命名冲突都在建目录前报告。
    for path in images:
        rel = path.relative_to(source_root)
        try:
            with Image.open(path) as image:
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('不接受多帧图片。')
                if image.size != (500, 700):
                    raise ValueError(f'必须为500×700，实际{image.size}；不缩放或补黑。')
                if image.mode not in PNG_MODES:
                    raise ValueError(f'不自动转换位深或颜色模式：{image.mode}')
                if 'exif' in image.info and int(image.getexif().get(274, 1)) != 1:
                    raise ValueError('初裁图必须已经归一化EXIF方向，不能在本步骤旋转。')
            # 按 normcase 后的绝对路径匹配父清单中的 initial 记录。
            parent = parent_records.get(os.path.normcase(str(path.resolve())))
            if manifest_path is not None and parent is None:
                raise ValueError('在指定manifest中找不到对应的initial记录。')
            # 统一小写化后查重：不同扩展名的同主名图片会生成相同的输出文件名。
            for tag in QUADRANTS:
                target = (rel.parent / f'{rel.stem}_{tag}.png').as_posix().casefold()
                if target in targets:
                    raise ValueError('输出文件名冲突，请先消除同目录同主名的不同格式图片。')
                targets.add(target)
        except (OSError, ValueError) as exc:
            raise ValueError(f'{path}: {exc}') from exc
        jobs.append((path, rel, parent))
    print(f'输入{len(jobs)}张500×700，计划四象限{len(jobs)*4}张250×350；格式PNG。', flush=True)
    if args.dry_run:
        # 只显示计划，不建输出目录、不写任何文件。
        for _, rel, _ in jobs:
            print(f'{rel}: 4张；xyxy={list(QUADRANTS.items())}')
        return {'dry_run': True, 'planned_sources': len(jobs), 'planned_quadrants': len(jobs)*4}
    # 预检全部通过后才检查空间并创建输出目录；目录已存在会在此失败。
    budget = check_space(output, len(jobs), args.preview)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        'tool': 'split_img.py', 'version': VERSION, 'manifest_schema_version': 3,
        'started_at': datetime.now().astimezone().isoformat(), 'input': str(source), 'output': str(output),
        'parent_manifest': str(manifest_path) if manifest_path else None,
        'parameters': {'input_size': [500, 700], 'output_size': [250, 350], 'format': 'PNG',
                       'limit': args.limit, 'preview': args.preview, 'preview_count': 1 if args.preview else 0,
                       'pixel_verification': 'all_output_files', 'quality_filtering': False},
        'planned_sources': len(jobs), 'planned_quadrants': len(jobs)*4,
        'original_source_groups': len({p['source_id'] for _, _, p in jobs if p}),
        'disk_space_estimate_bytes': budget, 'completed_sources': 0, 'quadrants': 0,
        'pixels_verified': 0, 'source_integrity': [], 'reference_integrity': [], 'errors': [],
    }
    # manifest.jsonl 与图像同步写出：每张象限输出各占一行 JSON 记录。
    with (output / 'manifest.jsonl').open('x', encoding='utf-8') as stream:
        for index, (path, rel, parent) in enumerate(jobs):
            digest, before, failure = None, None, None
            try:
                before = file_state(path)
                # 预检与正式处理之间初裁图可能被改动，先比对快照再继续。
                if before != inventory[rel.as_posix()]:
                    raise RuntimeError('初裁图在预检后发生变化。')
                digest = sha256_file(path)
                with Image.open(path) as image:
                    image.load()
                    if image.size != (500, 700) or getattr(image, 'n_frames', 1) != 1:
                        raise RuntimeError('图片尺寸或帧数在预检后发生变化。')
                    for tag, box in QUADRANTS.items():
                        qrel = rel.parent / f'{rel.stem}_{tag}.png'
                        with image.crop(box) as quadrant:
                            save_verified_png(quadrant, output / qrel)
                        summary['quadrants'] += 1
                        summary['pixels_verified'] += 1
                        stream.write(json.dumps(quadrant_record(parent, path, rel, digest, tag, box, qrel),
                                                ensure_ascii=False) + '\n')
                    # 独立预览只为排序第一张生成一次。
                    if args.preview and index == 0:
                        save_preview(image, output / 'previews' / 'quadrant_example.png')
            except Exception as exc:
                failure = str(exc)
            # 无论是否出错，只要已计算过摘要，就记录初裁图处理前后的完整性。
            if digest is not None:
                try:
                    after, after_hash = file_state(path), sha256_file(path)
                    unchanged = before == after and digest == after_hash
                    summary['source_integrity'].append({'path': str(path), 'sha256_before': digest,
                        'sha256_after': after_hash, 'size_before': before['size'], 'size_after': after['size'],
                        'mtime_ns_before': before['mtime_ns'], 'mtime_ns_after': after['mtime_ns'],
                        'unchanged': unchanged})
                    if not unchanged:
                        failure = failure or '初裁图前后摘要、大小或修改时间不同。'
                except OSError as exc:
                    failure = failure or f'初裁图完整性检查失败：{exc}'
            if failure:
                # 首个失败即停止后续图片：保留已写结果与错误记录，不删除、不覆盖重跑。
                summary['errors'].append({'source': str(path), 'error': failure})
                break
            summary['completed_sources'] += 1
            if (index+1) % 100 == 0 or index == len(jobs)-1:
                print(f'[{index+1}/{len(jobs)}] 四象限{summary["quadrants"]}，逐片像素已核对，输入unchanged=true', flush=True)
    # 收尾核对：整个输入树与父清单相对照，任何变化都记入 errors。
    try:
        after_inventory = {p.relative_to(source_root).as_posix(): file_state(p) for p in discover_images(source)}
        differences = [key for key in sorted(inventory.keys() | after_inventory.keys())
                       if inventory.get(key) != after_inventory.get(key)]
        summary['input_tree_check'] = {'images_before': len(inventory), 'images_after': len(after_inventory),
                                       'metadata_differences': differences, 'unchanged': not differences}
        if differences:
            summary['errors'].append({'source': str(source), 'error': '输入图像目录发生变化。'})
        if manifest_path:
            after = {'path': str(manifest_path), **file_state(manifest_path), 'sha256': sha256_file(manifest_path)}
            unchanged = reference_before == after
            summary['reference_integrity'].append({'before': reference_before, 'after': after, 'unchanged': unchanged})
            if not unchanged:
                summary['errors'].append({'source': str(manifest_path), 'error': '父级manifest发生变化。'})
    except Exception as exc:
        summary['errors'].append({'source': str(source), 'error': f'最终完整性检查失败：{exc}'})
    summary['finished_at'] = datetime.now().astimezone().isoformat()
    # run_summary.json 是验收入口：数量对账、像素核验与完整性记录均以它为准。
    with (output / 'run_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f'完成{summary["completed_sources"]}/{len(jobs)}张，四象限{summary["quadrants"]}，错误{len(summary["errors"])}。', flush=True)
    print(f'另存位置：{output}', flush=True)
    return summary


def get_parser() -> argparse.ArgumentParser:
    """命令行参数定义；各参数的详细语义见同目录 README 第 10 节。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='500×700初裁图文件或目录')
    parser.add_argument('--output', type=Path, required=True, help='尚不存在的新输出目录，保留输入相对层级')
    parser.add_argument('--manifest', type=Path, help='上一轮core_crop的manifest.jsonl，用于继承原图来源')
    parser.add_argument('--limit', type=int, default=0, help='相对路径排序前N张；0为全部')
    parser.add_argument('--dry-run', action='store_true', help='只预检和显示计划，不建目录')
    parser.add_argument('--preview', action='store_true', help='仅为排序第一张输入生成独立示意预览')
    return parser


def main(argv: list[str] | None = None) -> int:
    """退出码：0 成功或 dry-run；1 正式处理阶段有失败记录；2 参数/路径/预检错误。"""
    args = get_parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 2
    return 1 if summary.get('errors') else 0


if __name__ == '__main__':
    raise SystemExit(main())
