from isl_static_prototype.data_pipeline import PROJECT_ROOT

                                                              
                            
 
                      
                                                       
 
        
                          
                               
 
         
 
                  
                 
               
                
 
              
                          
                           
 
            
                    
 
                                                              
                
           
                       
                             
 
                                
 
                            
                   
                                  
                                 
                                      
 
                                   
 
                                              
                                                              

from pathlib import Path
import json

import numpy as np
import pandas as pd


                                                              
       
                                                              

FEATURE_ROOT = (
    PROJECT_ROOT / "data" / "features"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT / "artifacts"
)

CONFIG_ROOT = (
    PROJECT_ROOT / "configs"
)

FEATURE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

ARTIFACT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

CONFIG_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


                                                              
               
                                                              

CLASS_NAMES = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes",
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


                                                              
                        
                                                              

WRIST = 0

                                            
PALM_MCP = [
    5,              
    9,               
    13,            
    17,             
]

                                                 
PALM_CENTER_POINTS = PALM_MCP

                  
FINGERTIPS = [
    4,          
    8,          
    12,          
    16,        
    20,         
]

                                                              
                                
 
                                          
                                                              

ANGLE_TRIPLETS = [
           
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),

           
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),

            
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),

          
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),

           
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]


                                                              
                  
                                                              

def safe_norm(vector):
    """
    Euclidean norm of the final dimension.
    """
    return np.linalg.norm(
        vector,
        axis=-1
    )


def joint_angle(points, a, b, c):
    """
    Angle ABC in radians.

    points:
        shape (21, 3)

    Returns:
        scalar angle in [0, pi]
    """

    v1 = points[a] - points[b]
    v2 = points[c] - points[b]

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0

    cosine = np.dot(v1, v2) / (
        n1 * n2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )


def pairwise_distances(points):
    """
    Euclidean distances between all pairs
    of a small set of points.

    Returns upper-triangle distances only.
    """

    distances = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(
                np.linalg.norm(
                    points[i] - points[j]
                )
            )

    return np.asarray(
        distances,
        dtype=np.float32
    )


                                                              
                        
                                                              

def normalize_landmarks(
    landmarks,
    epsilon=1e-6
):
    """
    Canonical hand normalization.

    1. Translate the hand so the wrist is the origin.
    2. Estimate hand/palm scale using the mean 2D distance
       from wrist to the four MCP landmarks.
    3. Divide x, y and z by the same scale.

    Input:
        (21, 3)

    Output:
        normalized_landmarks (21, 3)
        scale
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected (21, 3), got {landmarks.shape}"
        )

                                                              
                     
                                                              

    centered = (
        landmarks
        - landmarks[WRIST]
    )

                                                              
                
     
                                                       
                                                             
                                                              

    wrist_xy = centered[
        WRIST,
        :2
    ]

    mcp_xy = centered[
        PALM_MCP,
        :2
    ]

    palm_distances = np.linalg.norm(
        mcp_xy - wrist_xy,
        axis=1
    )

    scale = float(
        np.mean(palm_distances)
    )

    if scale < epsilon:
        raise ValueError(
            "Degenerate hand: palm scale too small."
        )

    normalized = (
        centered / scale
    )

    return (
        normalized.astype(np.float32),
        scale
    )


                                                              
                      
                                                              

def build_features(landmarks):
    """
    Convert raw MediaPipe landmarks into:

        node_features:
            (21, 5)

        global_features:
            (35,)
    """

    normalized, scale = normalize_landmarks(
        landmarks
    )

                                                              
                       
                        
                                                              

    xyz = normalized

                                                              
                     
                         
                                                              

    wrist_dist = safe_norm(
        normalized - normalized[WRIST]
    )

                                                              
                 
                                                              

    palm_center = np.mean(
        normalized[
            PALM_CENTER_POINTS
        ],
        axis=0
    )

                                                              
                     
                               
                                                              

    palm_dist = safe_norm(
        normalized - palm_center
    )

                                                              
                         
     
                     
                                                              

    node_features = np.concatenate(
        [
            xyz,
            wrist_dist[:, None],
            palm_dist[:, None],
        ],
        axis=1
    ).astype(np.float32)

                                                              
                     
                                                              

    global_features = []

                                                              
                     
                                                              

    for a, b, c in ANGLE_TRIPLETS:

        global_features.append(
            joint_angle(
                normalized,
                a,
                b,
                c
            )
        )

                                                              
                                   
                                                              

    for tip in FINGERTIPS:

        distance = np.linalg.norm(
            normalized[tip]
            - normalized[WRIST]
        )

        global_features.append(
            float(distance)
        )

                                                              
                                  
                                                              

    for tip in FINGERTIPS:

        distance = np.linalg.norm(
            normalized[tip]
            - palm_center
        )

        global_features.append(
            float(distance)
        )

                                                              
                                        
                                                              

    fingertip_points = normalized[
        FINGERTIPS
    ]

    global_features.extend(
        pairwise_distances(
            fingertip_points
        ).tolist()
    )

    global_features = np.asarray(
        global_features,
        dtype=np.float32
    )

    if node_features.shape != (21, 5):
        raise RuntimeError(
            f"Unexpected node shape: "
            f"{node_features.shape}"
        )

    if global_features.shape != (35,):
        raise RuntimeError(
            f"Unexpected global feature shape: "
            f"{global_features.shape}"
        )

    return (
        node_features,
        global_features,
        scale
    )


                                                              
                        
                                                              

metadata_path = (
    PROJECT_ROOT
    / "data"
    / "landmarks"
    / "metadata.csv"
)

metadata_df = pd.read_csv(
    metadata_path
)

                                       
usable = metadata_df[
    metadata_df["detection_success"]
].copy()

print(
    f"Usable landmark samples: "
    f"{len(usable)} / {len(metadata_df)}"
)

print()

for split in ["train", "val", "test"]:

    count = int(
        (
            usable["split"] == split
        ).sum()
    )

    print(
        f"{split:>5}: {count:4d}"
    )

print()


                                                              
                    
                                                              

feature_stats = {}

for split in ["train", "val", "test"]:

    split_df = usable[
        usable["split"] == split
    ].copy()

    node_features_all = []
    global_features_all = []
    labels_all = []
    filenames_all = []
    scales_all = []

    print("=" * 70)
    print(f"BUILDING FEATURES: {split.upper()}")
    print("=" * 70)

    for row in split_df.to_dict(
        orient="records"
    ):

        landmark_path = Path(
            row["landmark_path"]
        )

                                                    
        absolute_landmark_path = (
            PROJECT_ROOT / landmark_path
        )

        landmarks = np.load(
            absolute_landmark_path
        )

        node_features, global_features, scale = (
            build_features(
                landmarks
            )
        )

        node_features_all.append(
            node_features
        )

        global_features_all.append(
            global_features
        )

        labels_all.append(
            CLASS_TO_ID[row["class"]]
        )

        filenames_all.append(
            row["filename"]
        )

        scales_all.append(
            scale
        )

    X_node = np.stack(
        node_features_all
    ).astype(np.float32)

    X_global = np.stack(
        global_features_all
    ).astype(np.float32)

    y = np.asarray(
        labels_all,
        dtype=np.int64
    )

    scales = np.asarray(
        scales_all,
        dtype=np.float32
    )

    filenames = np.asarray(
        filenames_all,
        dtype=str
    )

                                                              
                   
                                                              

    assert X_node.ndim == 3
    assert X_node.shape[1:] == (21, 5)

    assert X_global.ndim == 2
    assert X_global.shape[1] == 35

    assert y.ndim == 1
    assert len(y) == len(X_node)

    assert np.isfinite(X_node).all()
    assert np.isfinite(X_global).all()

                                                              
          
                                                              

    output_path = (
        FEATURE_ROOT
        / f"{split}.npz"
    )

    np.savez_compressed(
        output_path,
        X_node=X_node,
        X_global=X_global,
        y=y,
        filenames=filenames,
        scales=scales,
    )

    feature_stats[split] = {
        "samples": int(len(y)),
        "node_shape": list(X_node.shape),
        "global_shape": list(X_global.shape),
        "class_counts": {
            CLASS_NAMES[class_id]: int(
                np.sum(y == class_id)
            )
            for class_id in range(
                len(CLASS_NAMES)
            )
        },
        "mean_palm_scale": float(
            np.mean(scales)
        ),
        "min_palm_scale": float(
            np.min(scales)
        ),
        "max_palm_scale": float(
            np.max(scales)
        ),
    }

    print(
        f"Samples:        {len(y)}"
    )

    print(
        f"Node features:  {X_node.shape}"
    )

    print(
        f"Global features:{X_global.shape}"
    )

    print(
        f"Saved:          {output_path}"
    )

    print()


                                                              
                                  
                                                              

normalization_config = {
    "coordinate_system": "wrist_centered",
    "scale_method": (
        "mean_2d_distance_from_wrist_to_palm_MCPs"
    ),
    "wrist_index": WRIST,
    "palm_mcp_indices": PALM_MCP,
    "palm_center_indices": PALM_CENTER_POINTS,
    "epsilon": 1e-6,
    "z_normalized_by_same_scale": True,
}


normalization_path = (
    ARTIFACT_ROOT
    / "normalization.json"
)

with open(
    normalization_path,
    "w"
) as f:

    json.dump(
        normalization_config,
        f,
        indent=2
    )


                                                              
                            
                                                              

feature_yaml = f"""# ISL Static Prototype
# Feature configuration

node_features:
  dimension: 5
  channels:
    - normalized_x
    - normalized_y
    - normalized_z
    - distance_from_wrist
    - distance_from_palm_center

global_features:
  dimension: 35
  groups:
    joint_angles: 15
    fingertip_to_wrist: 5
    fingertip_to_palm: 5
    fingertip_pairwise: 10

landmarks:
  count: 21

classes:
  count: 6
"""

feature_config_path = (
    ARTIFACT_ROOT
    / "feature_config.yaml"
)

with open(
    feature_config_path,
    "w"
) as f:

    f.write(feature_yaml)


                                                              
                          
                                                              

classes_yaml = """# ISL Static Prototype
# Class configuration

classes:
  0: Hello
  1: IloveYou
  2: No
  3: Please
  4: Thanks
  5: Yes

unknown:
  trained_class: false
  rejection_state: true
"""

classes_path = (
    CONFIG_ROOT
    / "classes.yaml"
)

with open(
    classes_path,
    "w"
) as f:

    f.write(classes_yaml)


                                                              
              
                                                              

print("=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

for split, stats in feature_stats.items():

    print(
        f"\n{split.upper()}"
    )

    print(
        f"  samples:        {stats['samples']}"
    )

    print(
        f"  node shape:     {stats['node_shape']}"
    )

    print(
        f"  global shape:   {stats['global_shape']}"
    )

    print(
        f"  mean scale:     "
        f"{stats['mean_palm_scale']:.5f}"
    )

    print(
        "  class counts:"
    )

    for class_name, count in (
        stats["class_counts"].items()
    ):

        print(
            f"    {class_name:10s}: {count}"
        )

print()
print(
    f"Normalization:   {normalization_path}"
)

print(
    f"Feature config:  {feature_config_path}"
)

print(
    f"Classes config:  {classes_path}"
)

print(
    f"Feature datasets:{FEATURE_ROOT}"
)
