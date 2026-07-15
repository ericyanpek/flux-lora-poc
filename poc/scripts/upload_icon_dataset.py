#!/usr/bin/env python3
"""
上传 icon 三组对照数据集到各自 S3 前缀,供 ctl.py train --layer icon-{pro,simple,bare} 使用。

三组图片完全相同,仅 caption(.txt)不同 —— 单变量对照实验(变量=caption 质量)。
源目录由 gen_icon_captions.py 生成。

Run: python3 upload_icon_dataset.py
"""
from pathlib import Path
import boto3
from config import REGION, BUCKET

BASE = Path(__file__).parent.parent / "dataset"
GROUPS = {
    "icon-pro":    (BASE / "icon-pro",    "datasets/icon-pro/"),
    "icon-simple": (BASE / "icon-simple", "datasets/icon-simple/"),
    "icon-bare":   (BASE / "icon-bare",   "datasets/icon-bare/"),
}
EXTS = {".png", ".txt"}


def main():
    s3 = boto3.client("s3", region_name=REGION)
    for name, (local, prefix) in GROUPS.items():
        if not local.exists():
            print(f"  ✗ {name}: 本地目录不存在 {local} —— 先跑 gen_icon_captions.py")
            continue
        files = [p for p in local.iterdir() if p.suffix.lower() in EXTS]
        pngs = sum(1 for p in files if p.suffix.lower() == ".png")
        txts = sum(1 for p in files if p.suffix.lower() == ".txt")
        for p in files:
            s3.upload_file(str(p), BUCKET, prefix + p.name)
        print(f"  ✓ {name}: {pngs} png + {txts} txt → s3://{BUCKET}/{prefix}")
    print("\n完成。训练:")
    print("  python3 ctl.py train --layer icon-pro")
    print("  python3 ctl.py train --layer icon-simple")
    print("  python3 ctl.py train --layer icon-bare")


if __name__ == "__main__":
    main()
