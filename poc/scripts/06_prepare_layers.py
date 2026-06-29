"""
从现有 18 张图生成两套 caption,上传到两个 S3 前缀,支撑分层 LoRA 训练。
- Style 层:详细描述内容(角色/物体/构图/颜色),不写画风 → 画风沉淀为 LoRA
- Character 层:稀疏 caption,删角色固有特征,只留 trigger → 角色身份焊进权重
Run: python3 06_prepare_layers.py
"""
import boto3
from pathlib import Path
from config import REGION, BUCKET, LOCAL_DATASET_PATH

LOCAL = Path(LOCAL_DATASET_PATH)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
STYLE_PREFIX = "datasets/slot-ip-v1-style/"
CHAR_PREFIX = "datasets/slot-ip-v1-char/"
STYLE_TRIGGER = "slotstyle"
CHAR_TRIGGER = "slotchar"


def style_caption(original: str) -> str:
    # Style 层:保留原详细描述(角色/构图/颜色),把风格词 SLOTIP 换成 style trigger
    body = original
    for token in ["SLOTIP style, ", "SLOTIP style ", "SLOTIP, ", "SLOTIP "]:
        if body.startswith(token):
            body = body[len(token):]
            break
    return f"{STYLE_TRIGGER}, {body}"


def char_caption(original: str) -> str:
    # Character 层:稀疏化 —— 只保留 trigger + 极简主体类别
    import re
    m = re.search(r"\b(skunk|pharaoh|mummy|elephant|gorilla|koi|horse|clown|chameleon|girl|skeleton|train|hero|zombie)\b",
                  original, re.IGNORECASE)
    if m:
        subject = m.group(1).lower()
    else:
        subject = "character"
        print(f"  WARN: no subject keyword matched, using fallback 'character'")
    return f"{CHAR_TRIGGER}, a {subject}"


def prepare():
    s3 = boto3.client("s3", region_name=REGION)
    imgs = [p for p in LOCAL.iterdir() if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(imgs)} images")
    for img in imgs:
        orig_txt = img.with_suffix(".txt")
        original = orig_txt.read_text().strip() if orig_txt.exists() else "a character"
        for prefix, capfn, label in [
            (STYLE_PREFIX, style_caption, "style"),
            (CHAR_PREFIX, char_caption, "char"),
        ]:
            cap = capfn(original)
            s3.upload_file(str(img), BUCKET, prefix + img.name)
            s3.put_object(Bucket=BUCKET, Key=prefix + img.stem + ".txt", Body=cap.encode())
        print(f"  {img.name}: style/char captions uploaded")
    print(f"\n✅ Style → s3://{BUCKET}/{STYLE_PREFIX}")
    print(f"✅ Char  → s3://{BUCKET}/{CHAR_PREFIX}")


if __name__ == "__main__":
    prepare()
