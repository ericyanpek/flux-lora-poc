#!/usr/bin/env python3
"""
中国风 Q版图标 —— 训练数据预处理。

策略(见对话讨论):
- 不裁边框(边框靠 caption 描述来解耦,而非物理裁掉)
- pad 成 1:1 正方形白底(RGBA 透明 → 白底合成,不缩放主体、只补白画布)
- 踢掉过小的图(鱼9 222x160,放大糊)
- 统一输出英文 ASCII 文件名(中文名在 ai-toolkit/S3 链路易出问题)
- 不统一放大到固定尺寸:保留各自原生分辨率(补白后),交给 ai-toolkit
  多分辨率 bucket [640,1024] 分桶,避免小图硬拉 1024 产生插值噪声

Run: python3 prep_icon_dataset.py
输入: /Users/yabolin/Downloads/中国风图标/Q版/*.png
输出: poc/dataset/icon/<ascii-name>.png  (1:1 白底,原生分辨率)
"""
from pathlib import Path
from PIL import Image

SRC = Path("/Users/yabolin/Downloads/中国风图标/Q版")
OUT = Path(__file__).parent.parent / "dataset" / "icon"

# 中文原名 -> ASCII 训练文件名(按题材+编号)。鱼9 标记 None = 踢掉。
NAME_MAP = {
    "杯壶类1 Q.png": "teapot_1",
    "杯壶类3 Q.png": "teacup_3",
    "鞭炮1 Q.png":   "firecracker_1",
    "鞭炮2 Q.png":   "firecracker_2",
    "鞭炮4 Q.png":   "firecracker_4",
    "鞭炮5 Q.png":   "firecracker_5",
    "灯笼1 Q.png":   "lantern_1",
    "灯笼5 Q.png":   "lantern_5",
    "灯笼7 Q.png":   "lantern_7",
    "鼎盆类2 Q.png": "cauldron_2",
    "蛤蟆1 Q.png":   "toad_1",
    "葫芦5 Q.png":   "gourd_5",
    "葫芦7 Q.png":   "gourd_7",
    "金币1 Q.png":   "goldcoin_1",
    "金币3 Q.png":   "goldcoin_3",
    "金鸡1 Q.png":   "goldrooster_1",
    "金鸡2 Q.png":   "goldrooster_2",
    "金元宝1 Q.png": "goldingot_1",
    "乐器类3 Q.png": "flute_3",
    "扇子1 Q.png":   "fan_1",
    "扇子3 Q.png":   "fan_3",
    "扇子4 Q.png":   "fan_4",
    "饰品1 Q.png":   "hairpin_1",
    "鱼12 Q.png":    "fish_12",
    "鱼9 Q.png":     None,   # 222x160 太小,放大糊 → 踢掉
    "中国结 Q.png":  "knot_1",
}


def pad_to_square_white(im: Image.Image) -> Image.Image:
    """RGBA 合成到白底,再补白成 1:1 正方形(不缩放主体)。"""
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    # 1) 先把透明区合成到白底
    white = Image.new("RGBA", im.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, im).convert("RGB")
    # 2) 补白成正方形(边长取长边)
    w, h = composited.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(composited, ((side - w) // 2, (side - h) // 2))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    for src_name, ascii_name in NAME_MAP.items():
        src_path = SRC / src_name
        if not src_path.exists():
            print(f"  ⚠ 源文件缺失: {src_name}")
            continue
        if ascii_name is None:
            dropped.append(src_name)
            print(f"  ✗ 踢掉: {src_name}")
            continue
        im = Image.open(src_path)
        w, h = im.size
        out = pad_to_square_white(im)
        out_path = OUT / f"{ascii_name}.png"
        out.save(out_path)
        kept.append((ascii_name, f"{w}x{h}", f"{out.size[0]}x{out.size[1]}"))
        print(f"  ✓ {src_name:22s} {w}x{h:<5} → {ascii_name}.png ({out.size[0]}²)")

    print(f"\n保留 {len(kept)} 张,踢掉 {len(dropped)} 张 → {OUT}")
    # 按补白后边长排序,给多分辨率 bucket 参考
    print("\n=== 补白后尺寸(供 [640,1024] 分桶参考) ===")
    for name, orig, sq in sorted(kept, key=lambda x: int(x[2].split("x")[0])):
        print(f"  {name:16s} 原 {orig:10s} 方 {sq}")


if __name__ == "__main__":
    main()
