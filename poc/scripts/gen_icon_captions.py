#!/usr/bin/env python3
"""
为 icon 数据集生成三套对照 caption,分别落到三个子目录(图片硬链接复用,仅 .txt 不同)。

三组单变量对照(变量 = caption 密度/质量):
  A 专业 (iconqstyle):  逐图看图定制的结构化 caption —— 物体+主色+辅助色/纹样+材质+边框。
                        写全"可变内容",故意不写画风特征(Q版/厚涂/高光/干净剪影),
                        让画风沉淀进 LoRA 权重。带框图显式写 frame → 边框被归一化为可控变量。
  B 一句话 (iconqsimple): 笼统一句 "a <color> <object>"。
  C 无 caption (iconqbare): 仅触发词,无任何描述。

Run: python3 gen_icon_captions.py
输入: poc/dataset/icon/*.png (prep_icon_dataset.py 的产物)
输出: poc/dataset/icon-pro/  icon-simple/  icon-bare/  (各含 png + txt)
"""
from pathlib import Path
import shutil

BASE = Path(__file__).parent.parent / "dataset"
SRC = BASE / "icon"

TRIG_PRO, TRIG_SIMPLE, TRIG_BARE = "iconqstyle", "iconqsimple", "iconqbare"

# 逐图: (专业 caption 主体部分, 一句话 caption 主体部分)
# 专业部分不含触发词(脚本统一加前缀);"within ... frame" / "isolated on white background" 显式描述边框状态。
ICONS = {
    "teapot_1": (
        "a teapot, magenta body with golden peony motif and gold lid, glossy ceramic material, dominant magenta with gold accents, isolated on white background",
        "a purple teapot"),
    "teacup_3": (
        "a teacup on a saucer filled with tea, green body with gold rim and lotus motif, glossy ceramic material, dominant green with gold accents, isolated on white background",
        "a green teacup"),
    "firecracker_1": (
        "a bundle of firecrackers with a hexagonal fu-character tag, blue crackers with gold caps, glossy material, dominant blue with gold accents, within a circular gold decorative frame",
        "blue firecrackers"),
    "firecracker_2": (
        "a scattered pile of firecrackers with a hexagonal fu-character tag, blue crackers with gold and red caps, glossy material, dominant blue with gold accents, within a blue square decorative frame",
        "blue firecrackers"),
    "firecracker_4": (
        "three bound firecrackers with a rope knot, green crackers with gold caps, glossy material, dominant green with gold accents, within a green rounded square decorative frame",
        "green firecrackers"),
    "firecracker_5": (
        "a scattered pile of firecrackers with a hexagonal fu-character tag, red crackers with gold caps, glossy material, dominant red with gold accents, isolated on white background",
        "red firecrackers"),
    "lantern_1": (
        "a pair of hanging lanterns with tassels, red body with gold ribs and caps, glossy material, dominant red with gold accents, isolated on white background",
        "red lanterns"),
    "lantern_5": (
        "a round hanging lantern, magenta body with gold cap and ribs, glossy material, dominant magenta with gold accents, within a purple rounded square decorative frame",
        "a purple lantern"),
    "lantern_7": (
        "a round hanging lantern, blue body with gold ribs and caps and a jade bead, glossy material, dominant blue with gold accents, isolated on white background",
        "a blue lantern"),
    "cauldron_2": (
        "a cauldron pot full of food with handles on a plate, red body with gold rim and flame motif, glossy ceramic material, dominant red with gold accents, isolated on white background",
        "a red pot"),
    "toad_1": (
        "a seated money toad on a pile of gold coins, blue body with a gold coin in its mouth, glossy material, dominant blue with gold accents, within a blue square decorative frame",
        "a blue toad"),
    "gourd_5": (
        "a calabash gourd with a rope and purple tassel, orange-red body with gold cloud motif, glossy material, dominant orange-red with gold and purple accents, isolated on white background",
        "an orange gourd"),
    "gourd_7": (
        "a calabash gourd lying down with a rope and beads, blue body with gold longevity emblem, glossy material, dominant blue with gold accents, within a blue rounded square decorative frame",
        "a blue gourd"),
    "goldcoin_1": (
        "three stacked round coins with a square hole and red ribbon, gold coins on a green disc, glossy metallic material, dominant gold with green and red accents, isolated on white background",
        "gold coins"),
    "goldcoin_3": (
        "three overlapping round coins with square holes and an orange ribbon, gold coins, glossy metallic material, dominant gold with orange accents, isolated on white background",
        "gold coins"),
    "goldrooster_1": (
        "a seated hen on a nest of gold coins, green body with gold and pink plumage details, glossy material, dominant green with gold and pink accents, within a green square decorative frame",
        "a green rooster"),
    "goldrooster_2": (
        "a seated hen on a nest of gold coins, golden body with pink plumage and gem details, glossy metallic material, dominant gold with pink accents, isolated on white background",
        "a gold rooster"),
    "goldingot_1": (
        "a large gold ingot on a pile of gold coins, golden ingot with jade band, glossy metallic material, dominant gold with green accents, within a green square decorative frame",
        "a gold ingot"),
    "flute_3": (
        "a bamboo flute with a rope knot and tassel, teal-green flute with gold ends, glossy material, dominant green with gold accents, within a green octagonal decorative frame",
        "a green flute"),
    "fan_1": (
        "an open folding fan with a gold handle and tassel, magenta pleats with gold ribs, glossy material, dominant magenta with gold accents, within a circular gold decorative frame",
        "a purple fan"),
    "fan_3": (
        "an open folding fan with a gold handle and tassel, magenta pleats with gold ribs, glossy material, dominant magenta with gold accents, within a purple octagonal decorative frame",
        "a purple fan"),
    "fan_4": (
        "an open folding fan with a gold handle and tassel, green pleats with gold ribs, glossy material, dominant green with gold accents, isolated on white background",
        "a green fan"),
    "hairpin_1": (
        "a lotus flower hairpin with dangling gold ornaments, teal-blue petals with a gold centerpiece, glossy material, dominant teal-blue with gold accents, isolated on white background",
        "a blue hairpin"),
    "fish_12": (
        "a plump koi fish on a wave of gold, magenta body with gold fins on golden clouds, glossy material, dominant magenta with gold accents, within a purple square decorative frame",
        "a purple fish"),
    "knot_1": (
        "a chinese decorative knot with a rope tassel and gold clasp, blue rope with gold accents, glossy material, dominant blue with gold accents, within a blue octagonal decorative frame",
        "a blue chinese knot"),
}


def main():
    dirs = {
        "pro":    (BASE / "icon-pro", TRIG_PRO),
        "simple": (BASE / "icon-simple", TRIG_SIMPLE),
        "bare":   (BASE / "icon-bare", TRIG_BARE),
    }
    for d, _ in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    missing = [n for n in ICONS if not (SRC / f"{n}.png").exists()]
    if missing:
        raise SystemExit(f"缺图,先跑 prep_icon_dataset.py: {missing}")

    for name, (pro_body, simple_body) in ICONS.items():
        png = SRC / f"{name}.png"
        # 图片复制到三个目录(保证三组图完全一致)
        for key, (d, trig) in dirs.items():
            shutil.copy2(png, d / f"{name}.png")
            if key == "pro":
                cap = f"{trig}, {pro_body}"
            elif key == "simple":
                cap = f"{trig}, {simple_body}"
            else:  # bare: 仅触发词
                cap = trig
            (d / f"{name}.txt").write_text(cap + "\n")

    print(f"生成 {len(ICONS)} 张 × 3 组 caption:")
    for key, (d, _) in dirs.items():
        print(f"  {key:7s} → {d}")
    print("\n=== A 专业 caption 预览(全部) ===")
    for name, (pro, _) in ICONS.items():
        print(f"  [{name}] {TRIG_PRO}, {pro}")
    print("\n=== B 一句话 caption 预览(全部) ===")
    for name, (_, simple) in ICONS.items():
        print(f"  [{name}] {TRIG_SIMPLE}, {simple}")
    print(f"\n=== C 无 caption: 每张仅 '{TRIG_BARE}' ===")


if __name__ == "__main__":
    main()
