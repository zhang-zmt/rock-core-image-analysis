#!/usr/bin/env python3
"""岩芯原图只读裁切：central保留旧流程；full做500x700二维全覆盖。

full默认只生成初裁图，显式--quadrants才生成四象限；不进行质量过滤。
依赖：Python >= 3.10、Pillow。所有结果另存到尚不存在的新目录。
坐标基于EXIF方向校正后的原图，采用xyxy、右下边界不包含。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

VERSION = '2.0.0'
# 支持的图片扩展名；大小写不敏感，其他扩展名在目录扫描时跳过、单文件输入时报错。
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# Windows 下禁止写入的源数据目录，防止裁切结果污染原图或原工程目录。
PROTECTED_ROOTS = tuple(Path(p).resolve() for p in os.environ.get("ROCK_CORE_PROTECTED_ROOTS", "").split(os.pathsep) if p)
# 允许直接保存的像素模式；遇到其他模式（如 CMYK、浮点）报错而不自动转换位深。
PNG_MODES = {'1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I;16'}


def sha256_file(path: Path) -> str:
    """按 1 MiB 分块读取计算文件 SHA-256，避免大图一次性占用内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 目录之内（先解析真实路径再做前缀比较）。"""
    return path.resolve().is_relative_to(root.resolve())


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """输入输出路径合法性检查：输入必须存在，输出必须尚不存在，且不嵌套、不写入受保护目录。"""
    source = source.resolve(strict=True)
    output = output.resolve()
    if not source.is_file() and not source.is_dir():
        raise ValueError('输入必须是普通图片文件或目录。')
    if any(inside(output, root) for root in PROTECTED_ROOTS):
        raise ValueError('禁止写入受保护的目录，请指定新的输出目录。')
    if source.is_dir() and (inside(output, source) or inside(source, output)):
        raise ValueError('输入与输出目录不能重合，也不能互相嵌套。')
    if output.exists():
        raise ValueError(f'输出已存在，不覆盖任何已有结果，请换一个新目录：{output}')
    return source, output


def discover_images(source: Path) -> list[Path]:
    """递归收集图片：单文件直接返回；目录则按相对路径（casefold）排序，保证 --limit 结果稳定。"""
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'不支持的图片扩展名：{source.suffix}')
        return [source]
    found = []

    def raise_walk_error(error):
        raise error  # 遍历中任何读取错误都中止，不静默跳过。

    # followlinks=False 且剔除符号链接/目录联接，避免重复或越出源目录地遍历。
    for directory, subdirs, names in os.walk(source, followlinks=False, onerror=raise_walk_error):
        parent = Path(directory)
        subdirs[:] = [name for name in subdirs
                      if not (parent / name).is_symlink()
                      and not getattr(parent / name, 'is_junction', lambda: False)()]
        for name in names:
            path = parent / name
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink():
                found.append(path)
    return sorted(found, key=lambda p: p.relative_to(source).as_posix().casefold())


def validate_size(crop_width: int, crop_height: int, scan: str) -> None:
    """校验裁片尺寸：必须为正偶数（保证四象限等分）；full 模式锁定 500×700。"""
    if scan not in {'central', 'full'}:
        raise ValueError('scan只能为central或full。')
    if crop_width <= 0 or crop_height <= 0 or crop_width % 2 or crop_height % 2:
        raise ValueError('裁片宽高必须为正偶数，保证四象限尺寸相等。')
    if scan == 'full' and (crop_width, crop_height) != (500, 700):
        raise ValueError('full模式固定使用500×700；自定义尺寸仅适用于central模式。')


def axis_windows(start: int, stop: int, length: int, fill_tail: bool) -> list[dict]:
    """一维覆盖计划；overlap为与前一位置窗口的交叠长度，不是剩余边长。

    常规窗口按步长 length 从 start 起依次排列；当最后一个常规窗口没有到达
    stop - length（即边缘未被覆盖）且 fill_tail 为真时，追加一个贴边补尾窗口。
    补尾窗口与前窗的交叠 overlap = 前窗末端 - 本窗起点（如宽4096、窗500时为404）。
    """
    if length <= 0 or stop - start < length:
        raise ValueError('有效区小于窗口，不能缩放或补黑。')
    positions = list(range(start, stop - length + 1, length))
    regular_count = len(positions)
    # 末窗不足一整步时补一个贴边窗口，保证覆盖到 stop 边界（右/下边界不包含）。
    if fill_tail and positions[-1] != stop - length:
        positions.append(stop - length)
    return [{'start': value, 'tail': i >= regular_count,
             'overlap': max(0, positions[i - 1] + length - value) if i else 0}
            for i, value in enumerate(positions)]


def plan_windows(width: int, height: int, crop_width: int = 500,
                 crop_height: int = 700, *, keep_tail: bool = False,
                 roi: tuple[int, int, int, int] | None = None,
                 scan: str = 'central') -> list[dict]:
    """生成一张图的全部初裁窗口计划（含行列号、补尾标记与重叠度量）。

    full 模式在 ROI（默认整图）内做横纵二维滑窗，步长等于窗口尺寸，
    始终补齐右侧与下侧；central 模式不纵向扫描，仅在垂直方向取一条居中
    条带，水平行为与旧版保持一致。
    """
    validate_size(crop_width, crop_height, scan)
    left, top, right, bottom = roi if roi is not None else (0, 0, width, height)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f'ROI越界或为空：{(left, top, right, bottom)}，图片尺寸：{(width, height)}')
    if right - left < crop_width or bottom - top < crop_height:
        raise ValueError('图片或ROI小于初裁窗口；不自动放大或补黑，请核对输入。')
    # full始终补齐两个方向；central保留旧版--keep-tail的水平补窗含义。
    xs = axis_windows(left, right, crop_width, keep_tail or scan == 'full')
    if scan == 'full':
        ys = axis_windows(top, bottom, crop_height, True)
    else:
        # central：垂直方向只取正中央的一条窗口高条带，无重叠、无补窗。
        ys = [{'start': top + (bottom - top - crop_height) // 2, 'tail': False, 'overlap': 0}]
    result = []
    # 逐行、每行从左到右展开为最终窗口列表；重叠面积按横纵两条交叠带的并集计。
    for row, yw in enumerate(ys):
        for col, xw in enumerate(xs):
            x, y = xw['start'], yw['start']
            ox, oy = xw['overlap'], yw['overlap']
            result.append({
                'bbox': (x, y, x + crop_width, y + crop_height),
                'scan': scan, 'roi_on_source': (left, top, right, bottom),
                'grid_row': row, 'grid_col': col, 'grid_rows': len(ys), 'grid_cols': len(xs),
                'is_tail_window': xw['tail'] or yw['tail'],
                'tail_right': xw['tail'], 'tail_bottom': yw['tail'],
                'overlap_x_px': ox, 'overlap_y_px': oy,
                'overlap_area_px': ox * crop_height + oy * crop_width - ox * oy,
                'overlap_scope': 'earlier_initial_windows_of_same_source',
            })
    return result


def initial_boxes(width: int, height: int, crop_width: int = 500,
                  crop_height: int = 700, *, keep_tail: bool = False,
                  roi: tuple[int, int, int, int] | None = None,
                  scan: str = 'central') -> list[tuple[int, int, int, int]]:
    """保留原公共接口及默认central坐标；full通过新关键字选择。"""
    return [item['bbox'] for item in plan_windows(width, height, crop_width, crop_height,
            keep_tail=keep_tail, roi=roi, scan=scan)]


def quadrant_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """把 width×height 等分为四个象限，返回各象限在父图中的 xyxy 裁框（无重叠无间隙）。"""
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError('四象限切分的输入宽高必须为正偶数。')
    mx, my = width // 2, height // 2
    return {'top_left': (0, 0, mx, my), 'top_right': (mx, 0, width, my),
            'bot_left': (0, my, mx, height), 'bot_right': (mx, my, width, height)}


def save_png(image: Image.Image, target: Path) -> None:
    """用独占模式（xb）写出 PNG；目标已存在会直接失败，绝不覆盖。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        image.save(stream, format='PNG', compress_level=3)


def _save_central_preview(image: Image.Image, boxes: list[tuple[int, int, int, int]],
                 target: Path) -> None:
    """只在预览副本画线；初裁及四象限训练图上不画线、不加字。"""
    # 旧版 central 预览：左侧整图叠加青色初裁框，右侧为首窗及其四象限缩略图。
    canvas = Image.new('RGB', (1220, 940), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    overview = ImageOps.contain(image.convert('RGB'), (640, 590))
    canvas.paste(overview, (15, 45))
    scale_x, scale_y = overview.width / image.width, overview.height / image.height
    draw.text((15, 15), f'Source {image.width}x{image.height}; cyan = initial crops', fill='black')
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        rect = (15 + round(x0 * scale_x), 45 + round(y0 * scale_y),
                15 + round(x1 * scale_x) - 1, 45 + round(y1 * scale_y) - 1)
        draw.rectangle(rect, outline='#00ffff', width=2)
        draw.text((rect[0] + 2, rect[1] + 2), str(i), fill='#ff0066')
    first = image.crop(boxes[0])
    draw.text((680, 15), 'First initial crop', fill='black')
    thumb = ImageOps.contain(first.convert('RGB'), (245, 360))
    canvas.paste(thumb, (680, 45))
    draw.text((945, 15), 'Four quadrants', fill='black')
    for j, (name, box) in enumerate(quadrant_boxes(*first.size).items()):
        patch = first.crop(box)
        thumb = ImageOps.contain(patch.convert('RGB'), (120, 168))
        x, y = 945 + (j % 2) * 130, 45 + (j // 2) * 200
        draw.text((x, y), name, fill='black')
        canvas.paste(thumb, (x, y + 20))
        patch.close()
    first.close()
    draw.text((15, 652), f'Initial crops: {len(boxes)} total (showing first 8)', fill='black')
    for i, box in enumerate(boxes[:8]):
        crop = image.crop(box)
        thumb = ImageOps.contain(crop.convert('RGB'), (140, 210))
        x = 15 + i * 150
        draw.text((x, 679), str(i), fill='black')
        canvas.paste(thumb, (x, 700))
        crop.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        canvas.save(stream, format='PNG')
    canvas.close()


def save_preview(image: Image.Image, windows: list[dict], target: Path,
                 scan: str, quadrants_enabled: bool) -> None:
    """只在预览副本画框；full展示完整二维网格及四角裁片，不过滤任何裁片。"""
    if scan == 'central' and quadrants_enabled:
        _save_central_preview(image, [item['bbox'] for item in windows], target)
        return
    # full/未启用四象限的预览：左侧整图网格总览，右侧为网格四角裁片的放大图。
    canvas = Image.new('RGB', (1520, 1040), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    overview = ImageOps.contain(image.convert('RGB'), (900, 855))
    canvas.paste(overview, (20, 70))
    # 缩略图到原图坐标的换算比例，用于把裁框按原坐标画在缩略图上。
    sx, sy = overview.width / image.width, overview.height / image.height
    first = windows[0]
    draw.text((20, 15), f"{scan.upper()}  source={image.width}x{image.height}  "
              f"grid={first['grid_cols']} columns x {first['grid_rows']} rows  initial={len(windows)}", fill='black')
    draw.text((20, 38), 'cyan=regular; orange=right tail; yellow=bottom tail; magenta=both tails', fill='black')
    # 颜色编码：青=常规窗，橙=右补窗，黄=下补窗，品红=右下双补窗。
    for index, window in enumerate(windows):
        x0, y0, x1, y1 = window['bbox']
        color = '#ff00ff' if window['tail_right'] and window['tail_bottom'] else (
            '#ff9900' if window['tail_right'] else '#ffff00' if window['tail_bottom'] else '#00ffff')
        rect = (20 + round(x0 * sx), 70 + round(y0 * sy),
                20 + round(x1 * sx) - 1, 70 + round(y1 * sy) - 1)
        draw.rectangle(rect, outline=color, width=2)
        draw.text((rect[0] + 2, rect[1] + 2), f"r{window['grid_row']}c{window['grid_col']}", fill='#ff0055')
    # 完整网格的四角，去掉只有一行/列时重复的缩略图位置。
    cols = first['grid_cols']
    indices = list(dict.fromkeys([0, cols - 1, len(windows) - cols, len(windows) - 1]))
    if scan == 'central':
        indices = list(dict.fromkeys([0, len(windows) // 3, 2 * len(windows) // 3, len(windows) - 1]))
    for i, index in enumerate(indices):
        window = windows[index]
        x, y = 960 + (i % 2) * 270, 70 + (i // 2) * 420
        draw.text((x, y), f"r{window['grid_row']:03d} c{window['grid_col']:03d}  "
                  f"x={window['bbox'][0]} y={window['bbox'][1]}", fill='black')
        draw.text((x, y + 17), f"overlap x/y={window['overlap_x_px']}/{window['overlap_y_px']} px", fill='black')
        with image.crop(window['bbox']) as patch:
            thumb = ImageOps.contain(patch.convert('RGB'), (240, 336))
            canvas.paste(thumb, (x, y + 40))
    draw.text((20, 962), f"Quadrant files: {'enabled' if quadrants_enabled else 'disabled'}; "
              'thumbnails are previews only; no quality filtering.', fill='black')
    draw.text((20, 987), f"ROI xyxy={first['roi_on_source']}; every initial crop keeps its original pixels.", fill='black')
    save_png(canvas, target)
    canvas.close()


def make_stem(key: Path, index: int, window: dict) -> str:
    """生成初裁文件主名：central 沿用旧式序号命名，full 带行列号与坐标便于溯源。"""
    if window['scan'] == 'central':
        return f'{key.name}_{index:02d}'  # 保持旧文件名。
    x, y, _, _ = window['bbox']
    return f"{key.name}_r{window['grid_row']:03d}_c{window['grid_col']:03d}_x{x:06d}_y{y:06d}"


def quadrant_window_metadata(window: dict, qbox: tuple[int, int, int, int]) -> dict:
    """四象限记录保留父网格定位，但重叠像素按象限自身尺寸计算。"""
    qx0, qy0, qx1, qy1 = qbox
    # 父窗与前窗的交叠带位于父窗左/上边缘内侧，取象限与该带的交集长度，
    # 不直接照抄父窗整段重叠值。
    ox = max(0, min(qx1, window['overlap_x_px']) - qx0)
    oy = max(0, min(qy1, window['overlap_y_px']) - qy0)
    meta = {k: v for k, v in window.items() if k != 'bbox'}
    # tail_flags_scope=parent_initial_window：补尾标记沿用父初裁窗口，
    # 避免把父窗级补窗重叠错误解释成 250px 宽象限自身的重叠。
    meta.update(overlap_x_px=ox, overlap_y_px=oy,
                overlap_area_px=ox * (qy1 - qy0) + oy * (qx1 - qx0) - ox * oy,
                overlap_scope='earlier_quadrants_of_previous_initial_windows',
                tail_flags_scope='parent_initial_window', parent_window=window)
    return meta


def check_free_space(output: Path, initial: int, quadrants: int, width: int, height: int,
                     previews: int) -> int:
    """在创建输出目录前估算并检查磁盘空间，返回保守估算值（字节）。"""
    # 用每像素4字节及余量做保守估算，不依赖尚未确定的PNG压缩率。
    required = int((initial * width * height * 4 + quadrants * (width // 2) * (height // 2) * 4
                    + previews * 1520 * 1040 * 4) * 1.02) + 32 * 1024 * 1024
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < required:
        raise ValueError(f'输出盘空间不足：保守估算需{required / 1024**3:.2f} GiB，'
                         f'可用{free / 1024**3:.2f} GiB；请换空间充足的新目录。')
    return required


def run(args: argparse.Namespace) -> dict:
    """主流程：路径校验→全部原图预检→（dry-run 返回计划）→建目录→逐图裁切→完整性核对→写 run_summary。

    任一原图在正式处理阶段失败即中止后续处理，保留已写结果与错误记录；
    返回的 summary 由 main 换算成退出码。
    """
    source, output = validate_paths(args.input, args.output)
    # 兼容旧调用：未提供 --scan 时按 central；四象限默认随 scan（central 开 / full 关）。
    scan = getattr(args, 'scan', 'central')
    requested_quadrants = getattr(args, 'quadrants', None)
    enabled = requested_quadrants if requested_quadrants is not None else scan == 'central'
    validate_size(args.width, args.height, scan)
    if args.limit < 0:
        raise ValueError('--limit不能为负数。')
    images = discover_images(source)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError('没有找到支持的图片。')
    source_root = source if source.is_dir() else source.parent
    jobs, seen = [], set()
    # 全部所选原图先预检；任何几何/帧数/模式错误都在建目录前报告。
    for path in images:
        try:
            rel = path.relative_to(source_root)
            # key 在原文件名与扩展名之间插入 "__" 并保留原扩展名，
            # 防止同目录下同主名不同格式的两张图输出时互相覆盖。
            key = rel.with_name(path.stem + '__' + path.suffix.lower().lstrip('.'))
            if key.as_posix().casefold() in seen:
                raise ValueError(f'输出标识冲突：{rel}')
            seen.add(key.as_posix().casefold())
            before = path.stat()
            with Image.open(path) as image:
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('暂不支持多帧图片，不能静默只取第一帧。')
                if image.mode not in PNG_MODES:
                    raise ValueError(f'不自动转换颜色/位深模式：{image.mode}')
                raw_size = image.size
                # EXIF 274（Orientation）为 5-8 时宽高互换；所有裁框坐标均基于方向校正后的尺寸。
                orientation = int(image.getexif().get(274, 1))
                oriented_size = raw_size[::-1] if orientation in {5, 6, 7, 8} else raw_size
            windows = plan_windows(*oriented_size, args.width, args.height,
                                   keep_tail=args.keep_tail, roi=tuple(args.roi) if args.roi else None,
                                   scan=scan)
        except (OSError, ValueError) as exc:
            raise ValueError(f'{path}: {exc}') from exc
        jobs.append({'path': path, 'relative': rel, 'key': key, 'raw_size': raw_size,
                     'oriented_size': oriented_size, 'orientation': orientation, 'windows': windows,
                     'stat_size': before.st_size, 'stat_mtime_ns': before.st_mtime_ns})
    planned = sum(len(job['windows']) for job in jobs)
    planned_q = planned * 4 if enabled else 0
    print(f'模式 {scan}；输入 {len(jobs)} 张，计划初裁 {planned} 张、四象限 {planned_q} 张。', flush=True)
    if args.dry_run:
        # 只输出每张图的网格计划与覆盖范围，不建输出目录、不写任何文件。
        print('只预检数量与坐标，不创建输出目录。')
        sources = []
        for job in jobs:
            windows = job['windows'];first = windows[0]
            coords = [item['bbox'] for item in windows]
            coverage = (min(b[0] for b in coords), min(b[1] for b in coords),
                        max(b[2] for b in coords), max(b[3] for b in coords))
            print(f"{job['relative']}  size={job['oriented_size']}  "
                  f"{first['grid_cols']}列×{first['grid_rows']}行={len(windows)}片  "
                  f"覆盖xyxy={coverage}  首窗={coords[0]}  末窗={coords[-1]}")
            sources.append({'source_relpath': job['relative'].as_posix(), 'size': job['oriented_size'],
                            'grid_cols': first['grid_cols'], 'grid_rows': first['grid_rows'],
                            'initial': len(windows), 'coverage_xyxy': coverage})
        return {'dry_run': True, 'scan': scan, 'sources': len(jobs), 'initial': planned,
                'quadrants': planned_q, 'source_plans': sources}

    # 预检全部通过后才估算空间并创建输出目录；目录已存在会在此失败。
    budget = check_free_space(output, planned, planned_q, args.width, args.height,
                              len(jobs) if args.preview else 0)
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        'version': VERSION, 'manifest_schema_version': 2,
        'created_at': datetime.now().astimezone().isoformat(), 'input': str(source), 'output': str(output),
        'parameters': {'scan': scan, 'width': args.width, 'height': args.height, 'stride': args.width,
                       'stride_xy': [args.width, args.height if scan == 'full' else None],
                       'keep_tail': args.keep_tail, 'effective_keep_tail': args.keep_tail or scan == 'full',
                       'quadrants_requested': requested_quadrants, 'quadrants_enabled': enabled,
                       'roi': args.roi, 'format': 'PNG', 'limit': args.limit, 'preview': args.preview},
        'coordinate_system': 'EXIF方向校正后的原图，xyxy，右下边界不包含',
        'disk_space_estimate_bytes': budget,
        'planned_sources': len(jobs), 'planned_initial': planned, 'planned_quadrants': planned_q,
        'initial': 0, 'quadrants': 0, 'completed_sources': 0, 'source_integrity': [], 'sources': [], 'errors': [],
    }
    # manifest.jsonl 与图像同步写出：每张初裁/象限各占一行 JSON 记录。
    with (output / 'manifest.jsonl').open('x', encoding='utf-8') as manifest:
        for source_index, job in enumerate(jobs):
            path, key = job['path'], job['key'];before = None;digest = None;failure = None
            try:
                before = path.stat()
                # 预检与正式处理之间原图可能被改动，先用大小和修改时间快速把关。
                if (before.st_size, before.st_mtime_ns) != (job['stat_size'], job['stat_mtime_ns']):
                    raise RuntimeError('原图在预检之后发生变化，请重新核对输入版本。')
                digest = sha256_file(path)
                with Image.open(path) as opened:
                    opened.load()
                    # 按 EXIF 方向转正后再裁切，坐标与 plan_windows 的计划一致。
                    with ImageOps.exif_transpose(opened) as image:
                        if image.size != job['oriented_size']:
                            raise ValueError('解码尺寸与预检不一致。')
                        # common 为该原图所有裁片记录共享的来源元数据；source_id 即原图 SHA-256。
                        common = {'manifest_schema_version': 2, 'source_path': str(path),
                                  'source_relpath': job['relative'].as_posix(), 'source_sha256': digest,
                                  'source_id': digest, 'raw_size': job['raw_size'], 'oriented_size': image.size,
                                  'exif_orientation': job['orientation'],
                                  'coordinate_system': summary['coordinate_system']}
                        # 先逐窗写出初裁，再（如启用）由每张初裁继续切四象限。
                        for index, window in enumerate(job['windows']):
                            box = window['bbox'];name = make_stem(key, index, window)
                            rel = Path('01_initial') / key.parent / (name + '.png')
                            meta = {k: v for k, v in window.items() if k != 'bbox'}
                            with image.crop(box) as initial:
                                save_png(initial, output / rel)
                                summary['initial'] += 1
                                record = {**common, **meta, 'stage': 'initial', 'crop_index': index,
                                          'bbox_on_source': box, 'size': initial.size,
                                          'output_relpath': rel.as_posix()}
                                manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                                if enabled:
                                    # 象限坐标换算到方向校正后原图：父窗左上角 + 象限内偏移。
                                    for tag, qbox in quadrant_boxes(*initial.size).items():
                                        qrel = Path('02_quadrants') / key.parent / f'{name}_{tag}.png'
                                        with initial.crop(qbox) as quadrant:
                                            save_png(quadrant, output / qrel)
                                            summary['quadrants'] += 1
                                            absolute = (box[0] + qbox[0], box[1] + qbox[1],
                                                        box[0] + qbox[2], box[1] + qbox[3])
                                            record = {**common, **quadrant_window_metadata(window, qbox),
                                                      'stage': 'quadrant', 'crop_index': index, 'quadrant': tag,
                                                      'parent_relpath': rel.as_posix(), 'bbox_on_initial': qbox,
                                                      'bbox_on_source': absolute, 'size': quadrant.size,
                                                      'output_relpath': qrel.as_posix()}
                                            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if args.preview:
                            save_preview(image, job['windows'], output / 'previews' / key.parent / (key.name + '.png'),
                                         scan, enabled)
            except Exception as exc:
                failure = str(exc)
            # 即使写出失败，也尽可能记录输入原图的前后完整性。
            if before is not None and digest is not None:
                try:
                    after = path.stat();after_digest = sha256_file(path)
                    unchanged = ((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                                 and digest == after_digest)
                    summary['source_integrity'].append({'path': str(path), 'sha256': digest,
                        'sha256_after': after_digest, 'size_before': before.st_size, 'size_after': after.st_size,
                        'mtime_ns_before': before.st_mtime_ns, 'mtime_ns_after': after.st_mtime_ns,
                        'unchanged': unchanged})
                    if not unchanged:
                        failure = failure or '检测到原图变化，请停止并检查原图版本。'
                except OSError as exc:
                    failure = failure or f'原图完整性复查失败：{exc}'
            if failure:
                # 首个失败即停止后续原图：保留已写结果与错误记录，不删除、不覆盖重跑。
                summary['errors'].append({'source': str(path), 'error': failure})
                print(f'失败 {path}：{failure}', file=sys.stderr, flush=True)
                break
            summary['completed_sources'] += 1
            first = job['windows'][0]
            summary['sources'].append({'source_relpath': job['relative'].as_posix(),
                'oriented_size': job['oriented_size'], 'grid_cols': first['grid_cols'],
                'grid_rows': first['grid_rows'], 'initial': len(job['windows']),
                'quadrants': len(job['windows']) * 4 if enabled else 0})
            print(f"[{source_index + 1}/{len(jobs)}] {job['relative']}："
                  f"{first['grid_cols']}列×{first['grid_rows']}行={len(job['windows'])}，原图unchanged=true", flush=True)
    summary['finished_at'] = datetime.now().astimezone().isoformat()
    # run_summary.json 是验收入口：数量对账与 source_integrity 完整性记录均以它为准。
    with (output / 'run_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"结果：初裁 {summary['initial']}，四象限 {summary['quadrants']}，失败 {len(summary['errors'])}。", flush=True)
    print(f'另存位置：{output}', flush=True)
    return summary


def get_parser() -> argparse.ArgumentParser:
    """命令行参数定义；各参数的详细语义见同目录 README。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='一张图片或递归读取的图片目录')
    parser.add_argument('--output', type=Path, required=True, help='尚不存在的新输出目录，位于源目录之外')
    parser.add_argument('--scan', choices=['central', 'full'], default='central', help='默认central；full为500×700二维全覆盖')
    parser.add_argument('--width', type=int, default=500, help='初裁宽度；full固定500，central可自定义正偶数')
    parser.add_argument('--height', type=int, default=700, help='初裁高度；full固定700，central可自定义正偶数')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), help='方向校正后原图上的有效区；越界或过小报错')
    parser.add_argument('--keep-tail', action='store_true', help='central补右侧；full始终补右和下侧，此参数可显式写出')
    parser.add_argument('--quadrants', action=argparse.BooleanOptionalAction, default=None,
                        help='显式启用/禁用四象限；central默认启用，full默认禁用')
    parser.add_argument('--preview', action='store_true', help='另存网格及裁片缩略图预览，正式裁片不画框')
    parser.add_argument('--limit', type=int, default=0, help='仅处理相对路径排序前N张，0为全部')
    parser.add_argument('--dry-run', action='store_true', help='仅显示计划、覆盖范围和首末窗，不创建目录')
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
