from isl_static_prototype.data_pipeline import DATASET_ROOT, PROJECT_ROOT

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

metadata_path = PROJECT_ROOT / "data" / "landmarks" / "metadata.csv"
metadata_df = pd.read_csv(metadata_path)
failed_df = metadata_df[~metadata_df["detection_success"]].copy()

print(f"Failed images: {len(failed_df)}")
print()

display(failed_df[["split", "filename", "class", "error", "image_width", "image_height"]])

coco_lookup = {}

for split in ["train", "valid", "test"]:
    json_path = DATASET_ROOT / split / "_annotations.coco.json"

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_lookup = {image["file_name"]: image for image in coco["images"]}
    annotations_lookup = {}

    for ann in coco["annotations"]:
        annotations_lookup.setdefault(ann["image_id"], []).append(ann)

    coco_lookup[split] = {"images": image_lookup, "annotations": annotations_lookup}

for _, row in failed_df.iterrows():
    split = row["split"]
    filename = row["filename"]
    image_path = Path(row["source_path"])
    image = Image.open(image_path).convert("RGB")

    image_record = coco_lookup[split]["images"].get(filename)

    if image_record is None:
        print(f"No COCO image record for: {filename}")
        continue

    image_id = image_record["id"]
    annotations = coco_lookup[split]["annotations"].get(image_id, [])

    display_image = image.copy()
    draw = ImageDraw.Draw(display_image)

    print()
    print("=" * 80)
    print(filename)
    print("Class:", row["class"])
    print("Image size:", image.size)
    print("Annotations:", len(annotations))

    for ann in annotations:
        bbox = ann.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h

        draw.rectangle([x, y, x2, y2], outline="red", width=4)

        print("BBox:", f"x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")

    plt.figure(figsize=(7, 7))
    plt.imshow(display_image)
    plt.title(f"{row['class']} | MediaPipe: no_hand_detected")
    plt.axis("off")
    plt.show()
