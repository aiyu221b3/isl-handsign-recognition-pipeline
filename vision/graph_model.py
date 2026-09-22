from isl_static_prototype.data_pipeline import PROJECT_ROOT

                                                              
                       
 
                      
                                                
 
               
 
                                  
                 
                 
                            
                 
                 
                        
                          
                 
                 
                        
                          
                 
                 
                          
                 
                                
                                
                                
                                      
                                    
                                
                                
                        
                              
                        
                         
                        
                             
                        
                        
                           
 
                                       
                                                              

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


                                                              
               
                                                              

DEVICE = torch.device("cpu")

NUM_NODES = 21
NODE_FEATURES = 5
GLOBAL_FEATURES = 35
NUM_CLASSES = 6

GRAPH_HIDDEN = 48
CLASSIFIER_HIDDEN = 64

DROPOUT = 0.10


                                                              
                       
                                                              
 
                             
 
           
 
        
            
 
        
            
 
         
               
 
       
                
 
        
                
 
             
                    
                                    
                                                              

HAND_EDGES = [
           
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

           
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

            
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

          
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

           
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def build_normalized_adjacency(num_nodes=NUM_NODES):
    """
    Build a fixed anatomical adjacency matrix.

    Steps:
        1. Add undirected skeletal edges.
        2. Add self-loops.
        3. Apply symmetric degree normalization:

            A_hat = D^(-1/2) A D^(-1/2)

    Returns:
        Tensor of shape (21, 21)
    """

    adjacency = torch.zeros(
        num_nodes,
        num_nodes,
        dtype=torch.float32
    )

    for i, j in HAND_EDGES:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

                 
    adjacency += torch.eye(
        num_nodes,
        dtype=torch.float32
    )

    degree = adjacency.sum(dim=1)

    degree_inv_sqrt = torch.pow(
        degree,
        -0.5
    )

    degree_matrix = torch.diag(
        degree_inv_sqrt
    )

    normalized = (
        degree_matrix
        @ adjacency
        @ degree_matrix
    )

    return normalized


                                       
ADJACENCY = build_normalized_adjacency()


                                                              
                         
                                                              

class GraphLayer(nn.Module):
    """
    Lightweight graph layer:

        H' = A_hat H W + b

    followed by ReLU.

    This is intentionally implemented directly with PyTorch
    instead of using a graph-learning framework.
    """

    def __init__(
        self,
        in_features,
        out_features,
    ):
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
        """
        x:
            (batch, nodes, features)

        adjacency:
            (nodes, nodes)

        returns:
            (batch, nodes, out_features)
        """

                                         
        aggregated = torch.matmul(
            adjacency,
            x
        )

                                         
        return self.linear(
            aggregated
        )


                                                              
            
                                                              

class ISLGraphClassifier(nn.Module):
    """
    Lightweight six-class static ISL classifier.

    Inputs:
        node_features:
            (batch, 21, 5)

        global_features:
            (batch, 35)

    Output:
        logits:
            (batch, 6)
    """

    def __init__(
        self,
        node_features=NODE_FEATURES,
        global_features=GLOBAL_FEATURES,
        graph_hidden=GRAPH_HIDDEN,
        classifier_hidden=CLASSIFIER_HIDDEN,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.graph1 = GraphLayer(
            node_features,
            graph_hidden
        )

        self.graph2 = GraphLayer(
            graph_hidden,
            graph_hidden
        )

        self.dropout = nn.Dropout(
            dropout
        )

        combined_features = (
            graph_hidden
            + global_features
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                combined_features,
                classifier_hidden
            ),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                classifier_hidden,
                num_classes
            )
        )

                                               
                                  
                                    
                              
        self.register_buffer(
            "adjacency",
            ADJACENCY.clone()
        )

    def forward(
        self,
        node_features,
        global_features
    ):
        """
        Forward pass.

        node_features:
            (B, 21, 5)

        global_features:
            (B, 35)
        """

                                         
                       
                                         

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

        fused = self.dropout(
            fused
        )

                                         
                    
                                         

        logits = self.classifier(
            fused
        )

        return logits


                                                              
                
                                                              

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "features"
)


def load_feature_split(split):
    """
    Load one cached feature split.

    Returns:
        X_node
        X_global
        y
        filenames
    """

    path = (
        FEATURE_ROOT
        / f"{split}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature dataset: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False
    )

    X_node = torch.from_numpy(
        data["X_node"]
    ).float()

    X_global = torch.from_numpy(
        data["X_global"]
    ).float()

    y = torch.from_numpy(
        data["y"]
    ).long()

    filenames = data["filenames"]

    return (
        X_node,
        X_global,
        y,
        filenames
    )


                                                              
                                  
                                                              

X_node_train, X_global_train, y_train, train_names = (
    load_feature_split("train")
)

X_node_val, X_global_val, y_val, val_names = (
    load_feature_split("val")
)

X_node_test, X_global_test, y_test, test_names = (
    load_feature_split("test")
)


                                                              
                          
                                                              

assert X_node_train.shape == (
    722,
    21,
    5
)

assert X_global_train.shape == (
    722,
    35
)

assert y_train.shape == (
    722,
)

assert X_node_val.shape == (
    81,
    21,
    5
)

assert X_global_val.shape == (
    81,
    35
)

assert X_node_test.shape == (
    33,
    21,
    5
)

assert X_global_test.shape == (
    33,
    35
)


                                                              
                   
                                                              

model = ISLGraphClassifier(
    node_features=NODE_FEATURES,
    global_features=GLOBAL_FEATURES,
    graph_hidden=GRAPH_HIDDEN,
    classifier_hidden=CLASSIFIER_HIDDEN,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(DEVICE)


                                                              
                   
                                                              

test_nodes = X_node_train[
    :8
].to(DEVICE)

test_global = X_global_train[
    :8
].to(DEVICE)

test_labels = y_train[
    :8
].to(DEVICE)

assert test_nodes.shape == (
    8,
    21,
    5
)

assert test_global.shape == (
    8,
    35
)


                                                              
              
                                                              

model.eval()

with torch.no_grad():

    test_logits = model(
        test_nodes,
        test_global
    )

print("Forward pass:")
print(
    "  Input node shape:   ",
    tuple(test_nodes.shape)
)

print(
    "  Input global shape: ",
    tuple(test_global.shape)
)

print(
    "  Output logits:      ",
    tuple(test_logits.shape)
)


assert test_logits.shape == (
    8,
    6
)


                                                              
                  
                                                              

criterion = nn.CrossEntropyLoss()

loss = criterion(
    test_logits,
    test_labels
)

print()
print(
    "CrossEntropyLoss:",
    float(loss.item())
)

assert torch.isfinite(loss)


                                                              
                    
                                                              

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

optimizer.zero_grad(
    set_to_none=True
)

batch_logits = model(
    test_nodes,
    test_global
)

batch_loss = criterion(
    batch_logits,
    test_labels
)

batch_loss.backward()

optimizer.step()

print()
print(
    "One training batch:"
)

print(
    "  Loss before/after backward:",
    float(batch_loss.item())
)

print(
    "  Backpropagation:            OK"
)


                                                              
                         
                                                              

model.eval()

benchmark_nodes = X_node_train[
    :1
].to(DEVICE)

benchmark_global = X_global_train[
    :1
].to(DEVICE
)

         
with torch.no_grad():

    for _ in range(10):

        _ = model(
            benchmark_nodes,
            benchmark_global
        )


                  
import time

iterations = 200

start = time.perf_counter()

with torch.no_grad():

    for _ in range(iterations):

        _ = model(
            benchmark_nodes,
            benchmark_global
        )

elapsed = (
    time.perf_counter()
    - start
)

latency_ms = (
    elapsed
    / iterations
    * 1000
)

print()
print(
    "CPU inference benchmark:"
)

print(
    f"  Mean model latency: "
    f"{latency_ms:.3f} ms/image"
)

print(
    f"  Model-only FPS:     "
    f"{1000.0 / latency_ms:.1f}"
)


                                                              
                 
                                                              

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print(
    "Model size:"
)

print(
    f"  Total parameters:     "
    f"{total_parameters:,}"
)

print(
    f"  Trainable parameters: "
    f"{trainable_parameters:,}"
)


                                                              
                           
                                                              

print()
print("=" * 70)
print("GRAPH MODEL SANITY CHECK COMPLETE")
print("=" * 70)

print(
    f"Nodes:              {NUM_NODES}"
)

print(
    f"Node features:      {NODE_FEATURES}"
)

print(
    f"Graph hidden width: {GRAPH_HIDDEN}"
)

print(
    f"Global features:    {GLOBAL_FEATURES}"
)

print(
    f"Classifier width:   {CLASSIFIER_HIDDEN}"
)

print(
    f"Output classes:     {NUM_CLASSES}"
)

print(
    f"Parameters:         {trainable_parameters:,}"
)

print()
print(
    "✅ Tensor shapes valid"
)

print(
    "✅ Forward pass valid"
)

print(
    "✅ CrossEntropyLoss valid"
)

print(
    "✅ Backpropagation valid"
)

print(
    "✅ CPU inference valid"
)
