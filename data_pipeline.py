from IPython.display import display


import sys
from pathlib import Path

import numpy as np
import mediapipe as mp

print("Python:", sys.version.split()[0])
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)


                                                              
                                
                                                              

from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")

print("Kaggle input directories:")
for p in KAGGLE_INPUT.iterdir():
    print("  ", p)

                                                 
                                         
candidates = []

for train_dir in KAGGLE_INPUT.rglob("train"):
    if not train_dir.is_dir():
        continue

    parent = train_dir.parent

    if (
        (parent / "valid").is_dir()
        and (parent / "test").is_dir()
    ):
        candidates.append(parent)

print("\nCandidate dataset roots:")

for p in candidates:
    print("  ", p)

if len(candidates) == 0:
    print("\n❌ No train/valid/test structure found.")
    print("\nLet's inspect the Kaggle filesystem:")
    
    for path in KAGGLE_INPUT.rglob("*"):
        if path.is_dir():
            print(path)

    raise FileNotFoundError(
        "\nCould not find a directory containing "
        "train/, valid/, and test/."
    )

elif len(candidates) == 1:
    DATASET_ROOT = candidates[0]
    print("\n✅ Dataset root found:")
    print(DATASET_ROOT)

else:
    print("\n⚠️ Multiple dataset roots found.")
    print("Set DATASET_ROOT manually from the candidates above.")

                                                              
                                
                                                              

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    split_dir = DATASET_ROOT / split

    images = [
        p for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{split:>5}: {len(images):4d} images")

                          
    for p in images[:5]:
        print("       ", p.name)

    print()

                                                              
                  
                                                              

CLASS_NAMES = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes",
]

def filename_to_class(filename: str) -> str:
    """
    Infer the six-class label from the filename.

    The dataset does NOT use class directories.
    Labels are encoded in filenames.
    """

    name = Path(filename).stem.lower()

                                  
    normalized = (
        name
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if normalized.startswith("hello"):
        return "Hello"

    if normalized.startswith("iloveyou"):
        return "IloveYou"

    if normalized.startswith("no"):
        return "No"

    if normalized.startswith("please"):
        return "Please"

    if normalized.startswith("thanks"):
        return "Thanks"

    if normalized.startswith("yes"):
        return "Yes"

    raise ValueError(
        f"Could not infer class from filename: {filename}"
    )


                                                  
examples = [
    "Hello-81794bae-a6d1-11ec-9ec1.jpg",
    "I-Love-You_036_png_rf.2a29242.jpg",
    "IloveYou-f3bdd3ca-a6d1-11ec.jpg",
    "No_053_png.rf.ba9ec967.jpg",
    "Please-0fee219c-a6d2-11ec.jpg",
    "Thanks-12345.jpg",
    "Yes-abcdef.jpg",
]

for example in examples:
    print(f"{example:60} -> {filename_to_class(example)}")

                                                              
                                     
                                                              

import urllib.request

PROJECT_ROOT = Path("/kaggle/working/isl_glasses")
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)

if not MODEL_PATH.exists():
    print("Downloading Hand Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")
else:
    print("Hand Landmarker already downloaded.")

print(f"Model path: {MODEL_PATH}")
print(f"Model size: {MODEL_PATH.stat().st_size / (1024**2):.2f} MB")

                                                              
                                      
                                                              

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)

print("✅ Hand Landmarker initialized.")
print("Mode: IMAGE")
print("Maximum hands: 1")
print("Expected landmarks per hand: 21")

                                                              
                                                
                                                              

test_images = [
    p for p in (DATASET_ROOT / "test").iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if not test_images:
    raise FileNotFoundError("No test images found.")

TEST_IMAGE = test_images[0]

print("Smoke-test image:")
print(TEST_IMAGE)
print("Class:", filename_to_class(TEST_IMAGE.name))

                                                              
                      
                                                              

from PIL import Image

def detect_hand(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    result = hand_landmarker.detect(mp_image)

    print("Image:", image_path.name)
    print("Class:", filename_to_class(image_path.name))
    print("Image shape:", image_np.shape)
    print("Hands detected:", len(result.hand_landmarks))

    if len(result.hand_landmarks) == 0:
        print("⚠️ No hand detected.")
        return result, None

    landmarks = result.hand_landmarks[0]

    print("Landmarks detected:", len(landmarks))

    coords = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    print("Coordinate shape:", coords.shape)
    print("Wrist:", coords[0])

    assert coords.shape == (21, 3)

    print("✅ 21-landmark sanity check passed.")

    return result, coords


result, landmarks = detect_hand(TEST_IMAGE)

                                                              
                      
                                                              

import pandas as pd
from pathlib import Path

rows = []

for split in ["train", "valid", "test"]:

    split_dir = DATASET_ROOT / split

    for img_path in split_dir.iterdir():

        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rows.append({
            "split": split,
            "filename": img_path.name,
            "path": str(img_path),
            "class": filename_to_class(img_path.name)
        })

df = pd.DataFrame(rows)

print(df.head())
print("\nTotal images:", len(df))

                                                              
               
                                                              

counts = (
    df.groupby(["split", "class"])
      .size()
      .unstack(fill_value=0)
)

display(counts)

print("\nOverall class counts:")
display(df["class"].value_counts().sort_index())

import json

for split in ["train","valid","test"]:

    json_path = DATASET_ROOT / split / "_annotations.coco.json"

    with open(json_path) as f:
        coco = json.load(f)

    print(f"\n{split.upper()}")
    print("Images:", len(coco["images"]))
    print("Annotations:", len(coco["annotations"]))
    print("Categories:", [c["name"] for c in coco["categories"]])

                                                              
                         
                                                              

for split in ["train","valid","test"]:

    json_path = DATASET_ROOT / split / "_annotations.coco.json"

    with open(json_path) as f:
        coco = json.load(f)

    image_ids = {img["id"] for img in coco["images"]}
    annotated = {ann["image_id"] for ann in coco["annotations"]}

    missing = image_ids - annotated

    print(f"{split}: {len(missing)} images without annotations")

from sklearn.model_selection import train_test_split

dev = df[df["split"] != "test"].copy()

train_df, val_df = train_test_split(
    dev,
    test_size=0.10,
    stratify=dev["class"],
    random_state=42,
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(df[df.split=="test"]))
