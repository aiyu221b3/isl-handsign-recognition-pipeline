from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "isl_static_best.pt"
HAND_MODEL_PATH = ROOT / "hand_landmarker.task"

CLASS_NAMES = [
    "Hello",
    "IloveYou",
    "No",
    "Please",
    "Thanks",
    "Yes",
]

NUM_CLASSES = 6
GRAPH_HIDDEN = 48
CLASSIFIER_HIDDEN = 64
GLOBAL_FEATURES = 35
DEVICE = torch.device("cpu")
BOX_PADDING = 0.35
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_adjacency():
    adjacency = torch.zeros(
        21,
        21,
        dtype=torch.float32
    ) # building 21x21 matrix with float zeros everywhere
    for i, j in HAND_EDGES: # taking all hand edges & marking
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    adjacency += torch.eye(21) # adding with identity matrix
    degree = adjacency.sum(dim=1) # turn into row vector
    inv_sqrt = torch.pow(
        degree,
        -0.5
    ) # exponentiate
    degree_matrix = torch.diag(
        inv_sqrt
    ) # take row matrix elem -> diag here
    return (
        degree_matrix
        @ adjacency
        @ degree_matrix
    ) # matrix multiplication 

class GraphLayer(nn.Module): # make a class of the graph NN
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features
        )
    def forward(
        self,
        x,
        adjacency
    ):
        aggregated = torch.matmul(
            adjacency,
            x
        )
        return self.linear(
            aggregated
        )






class ISLGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph1 = GraphLayer(
            5,
            GRAPH_HIDDEN
        )
        self.graph2 = GraphLayer(
            GRAPH_HIDDEN,
            GRAPH_HIDDEN
        )
        self.dropout = nn.Dropout(0.10)
        self.classifier = nn.Sequential(
            nn.Linear(
                GRAPH_HIDDEN + GLOBAL_FEATURES,
                CLASSIFIER_HIDDEN
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                CLASSIFIER_HIDDEN,
                NUM_CLASSES
            )
        )
        self.register_buffer(
            "adjacency",
            build_adjacency()
        )

    def forward(
        self,
        node_features,
        global_features
    ):
        x = self.graph1(
            node_features,
            self.adjacency
        )
        x = F.relu(x)
        x = self.graph2(
            x,
            self.adjacency
        )
        x = F.relu(x)
        graph_embedding = x.mean(
            dim=1
        )
        fused = torch.cat(
            [
                graph_embedding,
                global_features
            ],
            dim=1
        )
        fused = self.dropout(fused)
        return self.classifier(fused)






WRIST = 0
PALM_MCP = [
    5, 9, 13, 17
]
FINGERTIPS = [
    4, 8, 12, 16, 20
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


def joint_angle(points, a, b, c):
    v1 = points[a] - points[b]
    v2 = points[c] - points[b]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cosine = np.dot(v1, v2) / (n1 * n2)
    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )
    return float(np.arccos(cosine))


def pairwise_distances(points):
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


def build_features(landmarks):
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
    mcp_xy = centered[
        PALM_MCP,
        :2
    ]
    scale = float(
        np.mean(
            np.linalg.norm(
                mcp_xy,
                axis=1
            )
        )
    )
    if scale < 1e-6:
        raise ValueError(
            "Degenerate hand scale."
        )
    normalized = (
        centered / scale
    ).astype(np.float32)
    wrist_dist = np.linalg.norm(
        normalized - normalized[WRIST],
        axis=1
    )
    palm_center = np.mean(
        normalized[PALM_MCP],
        axis=0
    )
    palm_dist = np.linalg.norm(
        normalized - palm_center,
        axis=1
    )
    node_features = np.concatenate(
        [
            normalized,
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
        global_features.append(
            float(
                np.linalg.norm(
                    normalized[tip]
                    - normalized[WRIST]
                )
            )
        )
    for tip in FINGERTIPS:
        global_features.append(
            float(
                np.linalg.norm(
                    normalized[tip]
                    - palm_center
                )
            )
        )
    global_features.extend(
        pairwise_distances(
            normalized[FINGERTIPS]
        ).tolist()
    )
    global_features = np.asarray(
        global_features,
        dtype=np.float32
    )
    return (
        node_features,
        global_features
    )



def get_hand_bbox(
    landmarks,
    frame_width,
    frame_height,
    padding=BOX_PADDING
):
    """
    Get an enlarged visual bounding box from the
    21 MediaPipe landmarks.
    The padding is proportional to the detected
    hand width/height and clipped to the frame.
    """
    xs = np.asarray(
        [lm.x for lm in landmarks]
    )
    ys = np.asarray(
        [lm.y for lm in landmarks]
    )
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    width = max_x - min_x
    height = max_y - min_y
    width = max(width, 0.05)
    height = max(height, 0.05)
    min_x -= width * padding
    max_x += width * padding
    min_y -= height * padding
    max_y += height * padding
    x1 = max(
        0,
        int(min_x * frame_width)
    )
    y1 = max(
        0,
        int(min_y * frame_height)
    )
    x2 = min(
        frame_width - 1,
        int(max_x * frame_width)
    )
    y2 = min(
        frame_height - 1,
        int(max_y * frame_height)
    )
    return x1, y1, x2, y2






print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = ISLGraphClassifier().to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint.get('epoch', '?')}"
)






base_options = python.BaseOptions(
    model_asset_path=str(
        HAND_MODEL_PATH
    )
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = (
    vision.HandLandmarker
    .create_from_options(options)
)

# esp32 connectionS
cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    cap.release()
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


print()
print("=" * 60)
print("LIVE ISL CAMERA DEMO")
print("=" * 60)
print()
print("Show one hand to the camera.")
print("Press Q or ESC to quit.")
print()
print("No hand → WAITING...")
print(
    f"Bounding-box padding → {BOX_PADDING:.0%}"
)
print()


last_timestamp = 0
fps_ema = None






while True:

    loop_start = time.perf_counter()

    ok, frame = cap.read()

    if not ok:
        print(
            "Could not read webcam frame."
        )
        break

    frame_height, frame_width = frame.shape[:2]

    
    
    

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    
    
    

    timestamp_ms = (
        time.monotonic_ns()
        // 1_000_000
    )

    if timestamp_ms <= last_timestamp:
        timestamp_ms = last_timestamp + 1

    last_timestamp = timestamp_ms

    
    
    

    result = hand_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    
    
    

    label = "WAITING..."
    confidence = None

    
    
    

    if len(result.hand_landmarks) > 0:

        hand = result.hand_landmarks[0]

        landmarks = np.asarray(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm in hand
            ],
            dtype=np.float32
        )

        
        
        

        x1, y1, x2, y2 = get_hand_bbox(
            hand,
            frame_width,
            frame_height
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        
        
        

        try:

            node_features, global_features = (
                build_features(
                    landmarks
                )
            )

            node_tensor = (
                torch.from_numpy(
                    node_features
                )
                .unsqueeze(0)
                .float()
            )

            global_tensor = (
                torch.from_numpy(
                    global_features
                )
                .unsqueeze(0)
                .float()
            )

            with torch.no_grad():

                logits = model(
                    node_tensor,
                    global_tensor
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                predicted_id = int(
                    torch.argmax(
                        probabilities,
                        dim=1
                    ).item()
                )

                confidence = float(
                    probabilities[
                        0,
                        predicted_id
                    ].item()
                )

            label = CLASS_NAMES[
                predicted_id
            ]

        except Exception as exc:

            label = "FEATURE ERROR"

            cv2.putText(
                frame,
                str(exc)[:70],
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        
        
        

        if confidence is not None:

            status_text = (
                f"{label}  {confidence:.1%}"
            )

        else:

            status_text = label

        text_y = max(
            30,
            y1 - 12
        )

        cv2.putText(
            frame,
            status_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    else:

        
        
        

        cv2.putText(
            frame,
            "WAITING...",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    
    
    

    loop_time = (
        time.perf_counter()
        - loop_start
    )

    current_fps = (
        1.0 / loop_time
        if loop_time > 0
        else 0.0
    )

    if fps_ema is None:

        fps_ema = current_fps

    else:

        fps_ema = (
            0.9 * fps_ema
            + 0.1 * current_fps
        )

    cv2.putText(
        frame,
        f"Pipeline FPS: {fps_ema:.1f}",
        (20, frame_height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Q / ESC = quit",
        (20, frame_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    
    
    

    cv2.imshow(
        "ISL Static Prototype",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key in (
        ord("q"),
        27
    ):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()

print()
print("Camera demo stopped.")
