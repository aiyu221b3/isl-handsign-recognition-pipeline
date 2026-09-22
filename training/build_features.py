from isl_static_prototype.data_pipeline import PROJECT_ROOT
from pathlib import Path
import json
import numpy as np
import pandas as pd
# first, we let locations
FEATURE_ROOT = PROJECT_ROOT / 'data' / 'features'
ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'
CONFIG_ROOT = PROJECT_ROOT / 'configs'
FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['Hello', 'IloveYou', 'No', 'Please', 'Thanks', 'Yes'] # current set of signs
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
WRIST = 0
PALM_MCP = [5, 9, 13, 17]
PALM_CENTER_POINTS = PALM_MCP
FINGERTIPS = [4, 8, 12, 16, 20]
ANGLE_TRIPLETS = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 5, 6), (5, 6, 7), (6, 7, 8), (0, 9, 10), (9, 10, 11), (10, 11, 12), (0, 13, 14), (13, 14, 15), (14, 15, 16), (0, 17, 18), (17, 18, 19), (18, 19, 20)]

def safe_norm(vector):
    return np.linalg.norm(vector, axis=-1)

def joint_angle(points, a, b, c):
    """ points: (21, 3), returns: [0, pi]"""
    v1 = points[a] - points[b]
    v2 = points[c] - points[b]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-08 or n2 < 1e-08:
        return 0.0
    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))

def pairwise_distances(points):
    """find distance b/w pairs"""
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(points[i] - points[j]))
    return np.asarray(distances, dtype=np.float32)

def normalize_landmarks(landmarks, epsilon=1e-06):
    """ wrist -> origin, find dist from O to landmarks, divide x, y, z.
    input: (21, 3), and output is normalized that """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError(f'Expected (21, 3), got {landmarks.shape}')
    centered = landmarks - landmarks[WRIST]
    wrist_xy = centered[WRIST, :2]
    mcp_xy = centered[PALM_MCP, :2]
    palm_distances = np.linalg.norm(mcp_xy - wrist_xy, axis=1)
    scale = float(np.mean(palm_distances))
    if scale < epsilon:
        raise ValueError('Degenerate hand: palm scale too small.')
    normalized = centered / scale
    return (normalized.astype(np.float32), scale)

def build_features(landmarks):
    """node (21, 5) & global (35,) features"""
    normalized, scale = normalize_landmarks(landmarks)
    xyz = normalized
    wrist_dist = safe_norm(normalized - normalized[WRIST])
    palm_center = np.mean(normalized[PALM_CENTER_POINTS], axis=0)
    palm_dist = safe_norm(normalized - palm_center)
    node_features = np.concatenate([xyz, wrist_dist[:, None], palm_dist[:, None]], axis=1).astype(np.float32)
    global_features = []
    for a, b, c in ANGLE_TRIPLETS:
        global_features.append(joint_angle(normalized, a, b, c))
    for tip in FINGERTIPS:
        distance = np.linalg.norm(normalized[tip] - normalized[WRIST])
        global_features.append(float(distance))
    for tip in FINGERTIPS:
        distance = np.linalg.norm(normalized[tip] - palm_center)
        global_features.append(float(distance))
    fingertip_points = normalized[FINGERTIPS]
    global_features.extend(pairwise_distances(fingertip_points).tolist())
    global_features = np.asarray(global_features, dtype=np.float32)
    if node_features.shape != (21, 5):
        raise RuntimeError(f'Unexpected node shape: {node_features.shape}')
    if global_features.shape != (35,):
        raise RuntimeError(f'Unexpected global feature shape: {global_features.shape}')
    return (node_features, global_features, scale)

metadata_path = PROJECT_ROOT / 'data' / 'landmarks' / 'metadata.csv'
metadata_df = pd.read_csv(metadata_path)
usable = metadata_df[metadata_df['detection_success']].copy()
print(f'Usable landmark samples: {len(usable)} / {len(metadata_df)}')
print()
for split in ['train', 'val', 'test']:
    count = int((usable['split'] == split).sum())
    print(f'{split:>5}: {count:4d}')
print()
feature_stats = {}
for split in ['train', 'val', 'test']:
    split_df = usable[usable['split'] == split].copy()
    node_features_all = []
    global_features_all = []
    labels_all = []
    filenames_all = []
    scales_all = []
    print('=' * 70)
    print(f'BUILDING FEATURES: {split.upper()}')
    print('=' * 70)
    for row in split_df.to_dict(orient='records'):
        landmark_path = Path(row['landmark_path'])
        absolute_landmark_path = PROJECT_ROOT / landmark_path
        landmarks = np.load(absolute_landmark_path)
        node_features, global_features, scale = build_features(landmarks)
        node_features_all.append(node_features)
        global_features_all.append(global_features)
        labels_all.append(CLASS_TO_ID[row['class']])
        filenames_all.append(row['filename'])
        scales_all.append(scale)
    X_node = np.stack(node_features_all).astype(np.float32)
    X_global = np.stack(global_features_all).astype(np.float32)
    y = np.asarray(labels_all, dtype=np.int64)
    scales = np.asarray(scales_all, dtype=np.float32)
    filenames = np.asarray(filenames_all, dtype=str)
    assert X_node.ndim == 3
    assert X_node.shape[1:] == (21, 5)
    assert X_global.ndim == 2
    assert X_global.shape[1] == 35
    assert y.ndim == 1
    assert len(y) == len(X_node)
    assert np.isfinite(X_node).all()
    assert np.isfinite(X_global).all()
    output_path = FEATURE_ROOT / f'{split}.npz'
    np.savez_compressed(output_path, X_node=X_node, X_global=X_global, y=y, filenames=filenames, scales=scales)
    feature_stats[split] = {'samples': int(len(y)), 'node_shape': list(X_node.shape), 'global_shape': list(X_global.shape), 'class_counts': {CLASS_NAMES[class_id]: int(np.sum(y == class_id)) for class_id in range(len(CLASS_NAMES))}, 'mean_palm_scale': float(np.mean(scales)), 'min_palm_scale': float(np.min(scales)), 'max_palm_scale': float(np.max(scales))}
    print(f'Samples:        {len(y)}')
    print(f'Node features:  {X_node.shape}')
    print(f'Global features:{X_global.shape}')
    print(f'Saved:          {output_path}')
    print()
normalization_config = {'coordinate_system': 'wrist_centered', 'scale_method': 'mean_2d_distance_from_wrist_to_palm_MCPs', 'wrist_index': WRIST, 'palm_mcp_indices': PALM_MCP, 'palm_center_indices': PALM_CENTER_POINTS, 'epsilon': 1e-06, 'z_normalized_by_same_scale': True}
normalization_path = ARTIFACT_ROOT / 'normalization.json'
with open(normalization_path, 'w') as f:
    json.dump(normalization_config, f, indent=2)
feature_yaml = f'# ISL Static Prototype\n# Feature configuration\n\nnode_features:\n  dimension: 5\n  channels:\n    - normalized_x\n    - normalized_y\n    - normalized_z\n    - distance_from_wrist\n    - distance_from_palm_center\n\nglobal_features:\n  dimension: 35\n  groups:\n    joint_angles: 15\n    fingertip_to_wrist: 5\n    fingertip_to_palm: 5\n    fingertip_pairwise: 10\n\nlandmarks:\n  count: 21\n\nclasses:\n  count: 6\n'
feature_config_path = ARTIFACT_ROOT / 'feature_config.yaml'
with open(feature_config_path, 'w') as f:
    f.write(feature_yaml)
classes_yaml = '# ISL Static Prototype\n# Class configuration\n\nclasses:\n  0: Hello\n  1: IloveYou\n  2: No\n  3: Please\n  4: Thanks\n  5: Yes\n\nunknown:\n  trained_class: false\n  rejection_state: true\n'
classes_path = CONFIG_ROOT / 'classes.yaml'
with open(classes_path, 'w') as f:
    f.write(classes_yaml)
print('=' * 70)
print('FEATURE ENGINEERING COMPLETE')
print('=' * 70)
for split, stats in feature_stats.items():
    print(f'\n{split.upper()}')
    print(f"  samples:        {stats['samples']}")
    print(f"  node shape:     {stats['node_shape']}")
    print(f"  global shape:   {stats['global_shape']}")
    print(f"  mean scale:     {stats['mean_palm_scale']:.5f}")
    print('  class counts:')
    for class_name, count in stats['class_counts'].items():
        print(f'    {class_name:10s}: {count}')
print()
print(f'Normalization:   {normalization_path}')
print(f'Feature config:  {feature_config_path}')
print(f'Classes config:  {classes_path}')
print(f'Feature datasets:{FEATURE_ROOT}')
