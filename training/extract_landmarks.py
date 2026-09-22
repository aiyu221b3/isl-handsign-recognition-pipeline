from isl_static_prototype.data_pipeline import (
    DATASET_ROOT,
    PROJECT_ROOT,
    df,
    train_df,
    val_df,
    hand_landmarker,
)

                                                              
                               
 
                      
                                                
 
           
                                       
                                                      
                                       
                                 
                                                      
                                                  
 
         
                                               
                   
                 
                  
                    
 
                                 
                       
                       
                                                              

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp


                                                              
               
                                                              

LANDMARK_ROOT = PROJECT_ROOT / "data" / "landmarks"

for split in ["train", "val", "test"]:
    (LANDMARK_ROOT / split).mkdir(
        parents=True,
        exist_ok=True
    )


                                                              
                              
                                                              

coco_lookup = {}

for source_split in ["train", "valid", "test"]:

    json_path = (
        DATASET_ROOT
        / source_split
        / "_annotations.coco.json"
    )

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_lookup = {
        image["file_name"]: image
        for image in coco["images"]
    }

    annotation_lookup = {}

    for annotation in coco["annotations"]:
        annotation_lookup.setdefault(
            annotation["image_id"],
            []
        ).append(annotation)

    coco_lookup[source_split] = {
        "images": image_lookup,
        "annotations": annotation_lookup,
    }


                                                              
                                     
                                                              

split_tables = {
    "train": train_df.copy(),
    "val": val_df.copy(),
    "test": df[df["split"] == "test"].copy(),
}

print("Extraction dataset:")

for split, table in split_tables.items():
    print(
        f"  {split:>5}: "
        f"{len(table):4d} images"
    )

print()


                                                              
         
                                                              

def get_original_coco_split(split: str) -> str:
    """
    Convert our working split names back to the original
    dataset split names used by the COCO files.

    train -> train
    val   -> train/valid depending on original source file

    Since train_df and val_df were created from the merged
    development set, use the source path to determine the
    original COCO directory.
    """
    return split


def find_coco_annotation(image_path: Path, split: str):
    """
    Find the original COCO annotation for an image.

    The image filename is unique in this dataset, so the
    original directory can be recovered from the source path.
    """

    original_split = image_path.parent.name

                                                           
                                                              
                 
    if original_split not in coco_lookup:
        return None

    image_record = coco_lookup[original_split]["images"].get(
        image_path.name
    )

    if image_record is None:
        return None

    annotations = coco_lookup[original_split]["annotations"].get(
        image_record["id"],
        []
    )

    if not annotations:
        return None

    return annotations[0]


def mediapipe_detect(image_np: np.ndarray):
    """
    Run MediaPipe on an RGB NumPy image.
    """

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    return hand_landmarker.detect(mp_image)


def landmarks_from_result(result):
    """
    Convert the first MediaPipe hand to a (21, 3) float32 array.
    Return None if no valid hand is present.
    """

    if len(result.hand_landmarks) == 0:
        return None

    hand = result.hand_landmarks[0]

    landmarks = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand
        ],
        dtype=np.float32
    )

    if landmarks.shape != (21, 3):
        return None

    return landmarks


                                                              
                         
                                                              

def extract_landmarks(image_path: Path):
    """
    Extraction pipeline:

        full image
            ↓
        MediaPipe
            ↓
        success? ── yes → cache
            │
            no
            ↓
        COCO bbox + padding
            ↓
        MediaPipe crop
            ↓
        remap landmarks to original coordinates
            ↓
        cache
    """

    try:

                                                              
                    
                                                              

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image_width, image_height = image.size

            image_np = np.asarray(
                image,
                dtype=np.uint8
            )

                                                              
                               
                                                              

        result = mediapipe_detect(image_np)

        landmarks = landmarks_from_result(result)

        if landmarks is not None:

            handedness = None
            handedness_score = None

            if (
                result.handedness
                and len(result.handedness[0]) > 0
            ):
                category = result.handedness[0][0]
                handedness = category.category_name
                handedness_score = float(category.score)

            return {
                "detection_success": True,
                "extraction_method": "full_image",
                "handedness": handedness,
                "handedness_score": handedness_score,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": None,
                "crop_y1": None,
                "crop_x2": None,
                "crop_y2": None,
                "error": None,
            }, landmarks

                                                              
                                   
                                                              

        annotation = find_coco_annotation(
            image_path,
            image_path.parent.name
        )

        if annotation is None:

            return {
                "detection_success": False,
                "extraction_method": "failed_no_annotation",
                "handedness": None,
                "handedness_score": None,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": None,
                "crop_y1": None,
                "crop_x2": None,
                "crop_y2": None,
                "error": "no_hand_detected_and_no_coco_bbox",
            }, None

        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:

            return {
                "detection_success": False,
                "extraction_method": "failed_invalid_bbox",
                "handedness": None,
                "handedness_score": None,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": None,
                "crop_y1": None,
                "crop_x2": None,
                "crop_y2": None,
                "error": "invalid_coco_bbox",
            }, None

        x, y, w, h = map(float, bbox)

                                                              
                       
                                                              

        if w <= 0 or h <= 0:

            return {
                "detection_success": False,
                "extraction_method": "failed_invalid_bbox",
                "handedness": None,
                "handedness_score": None,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": None,
                "crop_y1": None,
                "crop_x2": None,
                "crop_y2": None,
                "error": "non_positive_coco_bbox",
            }, None

                                                              
                                     
                                                              

        padding = 0.15

        x1 = max(
            0,
            int(np.floor(x - w * padding))
        )

        y1 = max(
            0,
            int(np.floor(y - h * padding))
        )

        x2 = min(
            image_width,
            int(np.ceil(x + w * (1.0 + padding)))
        )

        y2 = min(
            image_height,
            int(np.ceil(y + h * (1.0 + padding)))
        )

        if x2 <= x1 or y2 <= y1:

            return {
                "detection_success": False,
                "extraction_method": "failed_invalid_crop",
                "handedness": None,
                "handedness_score": None,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "error": "invalid_crop_dimensions",
            }, None

        crop = image_np[y1:y2, x1:x2]

                                                              
                               
                                                              

        crop_result = mediapipe_detect(crop)

        crop_landmarks = landmarks_from_result(
            crop_result
        )

        if crop_landmarks is None:

            return {
                "detection_success": False,
                "extraction_method": "full_and_crop_failed",
                "handedness": None,
                "handedness_score": None,
                "image_width": image_width,
                "image_height": image_height,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "error": "no_hand_detected",
            }, None

                                                              
                                                           
         
                                     
                                                 
         
                             
                                                     
         
                                                              
                                                         
                                                              

        crop_height, crop_width = crop.shape[:2]

        original_landmarks = crop_landmarks.copy()

        original_landmarks[:, 0] = (
            (
                x1
                + crop_landmarks[:, 0] * crop_width
            )
            / image_width
        )

        original_landmarks[:, 1] = (
            (
                y1
                + crop_landmarks[:, 1] * crop_height
            )
            / image_height
        )

                                                        
                                                         
                                                              
                 
        original_landmarks[:, 2] *= (
            crop_width / image_width
        )

                                                              
                    
                                                              

        handedness = None
        handedness_score = None

        if (
            crop_result.handedness
            and len(crop_result.handedness[0]) > 0
        ):

            category = crop_result.handedness[0][0]

            handedness = category.category_name
            handedness_score = float(category.score)

        return {
            "detection_success": True,
            "extraction_method": "coco_crop",
            "handedness": handedness,
            "handedness_score": handedness_score,
            "image_width": image_width,
            "image_height": image_height,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "error": None,
        }, original_landmarks

    except Exception as exc:

        return {
            "detection_success": False,
            "extraction_method": "exception",
            "handedness": None,
            "handedness_score": None,
            "image_width": None,
            "image_height": None,
            "crop_x1": None,
            "crop_y1": None,
            "crop_x2": None,
            "crop_y2": None,
            "error": f"{type(exc).__name__}: {exc}",
        }, None


                                                              
                     
                                                              

records = []

total_start = time.perf_counter()

for split, table in split_tables.items():

    print("=" * 70)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 70)

    split_start = time.perf_counter()

    success_count = 0
    failure_count = 0
    full_count = 0
    crop_count = 0

    rows = table.to_dict(
        orient="records"
    )

    for index, row in enumerate(
        rows,
        start=1
    ):

        image_path = Path(
            row["path"]
        )

        filename = image_path.name

        landmark_path = (
            LANDMARK_ROOT
            / split
            / f"{image_path.stem}.npy"
        )

        metadata, landmarks = extract_landmarks(
            image_path
        )

                                                              
                                   
                                                              

        if landmarks is not None:

            np.save(
                landmark_path,
                landmarks
            )

            success_count += 1

            if metadata["extraction_method"] == "full_image":
                full_count += 1

            elif metadata["extraction_method"] == "coco_crop":
                crop_count += 1

            stored_path = str(
                landmark_path.relative_to(
                    PROJECT_ROOT
                )
            )

        else:

            failure_count += 1
            stored_path = None

                                                              
                       
                                                              

        records.append({
            "split": split,
            "filename": filename,
            "class": row["class"],
            "source_path": str(image_path),
            "landmark_path": stored_path,
            "detection_success": metadata[
                "detection_success"
            ],
            "extraction_method": metadata[
                "extraction_method"
            ],
            "handedness": metadata[
                "handedness"
            ],
            "handedness_score": metadata[
                "handedness_score"
            ],
            "image_width": metadata[
                "image_width"
            ],
            "image_height": metadata[
                "image_height"
            ],
            "crop_x1": metadata[
                "crop_x1"
            ],
            "crop_y1": metadata[
                "crop_y1"
            ],
            "crop_x2": metadata[
                "crop_x2"
            ],
            "crop_y2": metadata[
                "crop_y2"
            ],
            "error": metadata[
                "error"
            ],
        })

                                                              
                  
                                                              

        if (
            index % 50 == 0
            or index == len(rows)
        ):

            elapsed = (
                time.perf_counter()
                - split_start
            )

            print(
                f"[{index:4d}/{len(rows):4d}] "
                f"full={full_count:4d} "
                f"crop={crop_count:3d} "
                f"failed={failure_count:3d} "
                f"time={elapsed:.1f}s"
            )


                                                              
               
                                                              

metadata_df = pd.DataFrame(records)

metadata_path = (
    LANDMARK_ROOT / "metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


                                                              
              
                                                              

total_elapsed = (
    time.perf_counter()
    - total_start
)

print()
print("=" * 70)
print("LANDMARK EXTRACTION COMPLETE")
print("=" * 70)

for split in ["train", "val", "test"]:

    subset = metadata_df[
        metadata_df["split"] == split
    ]

    successful = int(
        subset[
            "detection_success"
        ].sum()
    )

    failed = len(subset) - successful

    full = int(
        (
            subset["extraction_method"]
            == "full_image"
        ).sum()
    )

    crop = int(
        (
            subset["extraction_method"]
            == "coco_crop"
        ).sum()
    )

    print(
        f"{split:>5}: "
        f"{len(subset):4d} total | "
        f"{successful:4d} detected | "
        f"{failed:3d} failed | "
        f"{full:4d} full | "
        f"{crop:3d} crop"
    )

print()
print(
    f"Total images:   {len(metadata_df)}"
)

print(
    f"Successful:     "
    f"{int(metadata_df['detection_success'].sum())}"
)

print(
    f"Failed:         "
    f"{int((~metadata_df['detection_success']).sum())}"
)

print(
    f"Full-image:     "
    f"{int((metadata_df['extraction_method'] == 'full_image').sum())}"
)

print(
    f"COCO-crop:      "
    f"{int((metadata_df['extraction_method'] == 'coco_crop').sum())}"
)

print(
    f"Metadata:       {metadata_path}"
)

print(
    f"Runtime:        {total_elapsed:.1f}s"
)


                                                              
                
                                                              

failed_df = metadata_df[
    ~metadata_df["detection_success"]
].copy()

print()

if len(failed_df) == 0:

    print(
        "✅ All images produced valid 21×3 landmarks."
    )

else:

    print("=" * 70)
    print("REMAINING FAILED DETECTIONS")
    print("=" * 70)

    display(
        failed_df[
            [
                "split",
                "filename",
                "class",
                "extraction_method",
                "error",
            ]
        ]
    )


                                                              
                           
                                                              

print()
print("Extraction method summary:")

display(
    pd.crosstab(
        metadata_df["split"],
        metadata_df["extraction_method"]
    )
)
