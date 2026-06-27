"""
Copies images from a local folder to S3, auto-generates .txt caption files
for any images that don't already have one.
"""
import boto3
from pathlib import Path
from config import REGION, BUCKET, DATASET_PREFIX, TRIGGER_WORD, LOCAL_DATASET_PATH

LOCAL_DATASET = Path(LOCAL_DATASET_PATH)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def auto_caption(image_name: str, trigger_word: str) -> str:
    stem = Path(image_name).stem.replace("-", " ").replace("_", " ")
    return f"a character illustration in {trigger_word} style, {stem}"


def upload_dataset():
    s3 = boto3.client("s3", region_name=REGION)
    images = [p for p in LOCAL_DATASET.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(images)} images in {LOCAL_DATASET}")

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            caption = auto_caption(img_path.name, TRIGGER_WORD)
            txt_path.write_text(caption)
            print(f"  Auto-captioned: {img_path.name} -> \"{caption}\"")
        else:
            print(f"  Using existing caption: {txt_path.name}")

        s3_key = DATASET_PREFIX + img_path.name
        s3.upload_file(str(img_path), BUCKET, s3_key)
        print(f"  Uploaded: s3://{BUCKET}/{s3_key}")

        s3_txt_key = DATASET_PREFIX + txt_path.name
        s3.upload_file(str(txt_path), BUCKET, s3_txt_key)
        print(f"  Uploaded: s3://{BUCKET}/{s3_txt_key}")

    print(f"\n✅ Dataset uploaded to s3://{BUCKET}/{DATASET_PREFIX}")
    print(f"   {len(images)} image+caption pairs")
    return f"s3://{BUCKET}/{DATASET_PREFIX}"


if __name__ == "__main__":
    uri = upload_dataset()
    print(f"\nDataset S3 URI: {uri}")
