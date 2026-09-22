from isl_static_prototype.data_pipeline import PROJECT_ROOT
from isl_static_prototype.vision.graph_model import ISLGraphClassifier
from pathlib import Path
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cpu')
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 15
NUM_WORKERS = 0
MODEL_ROOT = PROJECT_ROOT / 'models'
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_ROOT / 'isl_static_best.pt'
LAST_MODEL_PATH = MODEL_ROOT / 'isl_static_last.pt'
CLASS_NAMES = ['Hello', 'IloveYou', 'No', 'Please', 'Thanks', 'Yes']
NUM_CLASSES = len(CLASS_NAMES)

class ISLFeatureDataset(Dataset):
  
    def __init__(self, feature_path):
        data = np.load(feature_path, allow_pickle=False)
        self.X_node = torch.from_numpy(data['X_node']).float()
        self.X_global = torch.from_numpy(data['X_global']).float()
        self.y = torch.from_numpy(data['y']).long()
        self.filenames = data['filenames']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X_node[index], self.X_global[index], self.y[index])
FEATURE_ROOT = PROJECT_ROOT / 'data' / 'features'
TRAIN_PATH = FEATURE_ROOT / 'train.npz'
VAL_PATH = FEATURE_ROOT / 'val.npz'
train_dataset = ISLFeatureDataset(TRAIN_PATH)
val_dataset = ISLFeatureDataset(VAL_PATH)
print('Training samples:', len(train_dataset))
print('Validation samples:', len(val_dataset))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
model = ISLGraphClassifier(node_features=21 * 0 + 5, global_features=35, graph_hidden=48, classifier_hidden=64, num_classes=NUM_CLASSES, dropout=0.1).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return {'accuracy': float(accuracy), 'macro_f1': float(macro_f1), 'precision': precision, 'recall': recall, 'confusion_matrix': cm}

def train_one_epoch():
    model.train()
    running_loss = 0.0
    sample_count = 0
    for X_node, X_global, y in train_loader:
        X_node = X_node.to(DEVICE)
        X_global = X_global.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_node, X_global)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
    return running_loss / sample_count

def validate():
    model.eval()
    running_loss = 0.0
    sample_count = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for X_node, X_global, y in val_loader:
            X_node = X_node.to(DEVICE)
            X_global = X_global.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(X_node, X_global)
            loss = criterion(logits, y)
            predictions = torch.argmax(logits, dim=1)
            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            all_targets.extend(y.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
    val_loss = running_loss / sample_count
    metrics = calculate_metrics(np.asarray(all_targets), np.asarray(all_predictions))
    return (val_loss, metrics)
history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_macro_f1': [], 'learning_rate': []}
best_f1 = -np.inf
best_epoch = 0
epochs_without_improvement = 0
training_start = time.perf_counter()
print()
print('=' * 70)
print('TRAINING')
print('=' * 70)
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()
    train_loss = train_one_epoch()
    val_loss, val_metrics = validate()
    val_accuracy = val_metrics['accuracy']
    val_f1 = val_metrics['macro_f1']
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]['lr']
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    history['val_macro_f1'].append(val_f1)
    history['learning_rate'].append(current_lr)
    epoch_time = time.perf_counter() - epoch_start
    improved = val_f1 > best_f1
    if improved:
        best_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_macro_f1': best_f1, 'class_names': CLASS_NAMES, 'config': {'graph_hidden': 48, 'classifier_hidden': 64, 'dropout': 0.1, 'num_classes': NUM_CLASSES, 'node_features': 5, 'global_features': 35, 'seed': SEED}}, BEST_MODEL_PATH)
    else:
        epochs_without_improvement += 1
    print(f'Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | val_f1={val_f1:.4f} | lr={current_lr:.2e} | time={epoch_time:.2f}s')
    if epochs_without_improvement >= PATIENCE:
        print()
        print(f'Early stopping at epoch {epoch}.')
        break
torch.save({'epoch': history['epoch'][-1], 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_macro_f1': best_f1, 'class_names': CLASS_NAMES, 'config': {'graph_hidden': 48, 'classifier_hidden': 64, 'dropout': 0.1, 'num_classes': NUM_CLASSES, 'node_features': 5, 'global_features': 35, 'seed': SEED}}, LAST_MODEL_PATH)
history_df = pd.DataFrame(history)
history_path = PROJECT_ROOT / 'artifacts' / 'training_history.csv'
history_df.to_csv(history_path, index=False)
best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()
final_val_loss, final_val_metrics = validate()
final_accuracy = final_val_metrics['accuracy']
final_macro_f1 = final_val_metrics['macro_f1']
final_precision = final_val_metrics['precision']
final_recall = final_val_metrics['recall']
final_cm = final_val_metrics['confusion_matrix']
total_training_time = time.perf_counter() - training_start
print()
print('=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Best epoch:          {best_epoch}')
print(f'Best validation F1:  {best_f1:.4f}')
print(f'Final val accuracy:   {final_accuracy:.4f}')
print(f'Final val macro F1:   {final_macro_f1:.4f}')
print(f'Final val loss:       {final_val_loss:.4f}')
print(f'Training time:        {total_training_time:.1f}s')
print()
print('Best checkpoint:')
print(BEST_MODEL_PATH)
print()
print('Last checkpoint:')
print(LAST_MODEL_PATH)
print()
print('=' * 70)
print('PER-CLASS VALIDATION METRICS')
print('=' * 70)
for class_id, class_name in enumerate(CLASS_NAMES):
    print(f'{class_name:10s} | precision={final_precision[class_id]:.4f} | recall={final_recall[class_id]:.4f} | F1={2 * final_precision[class_id] * final_recall[class_id] / max(final_precision[class_id] + final_recall[class_id], 1e-12):.4f}')
print()
print('=' * 70)
print('VALIDATION CONFUSION MATRIX')
print('=' * 70)
cm_df = pd.DataFrame(final_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
display(cm_df)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_df['epoch'], history_df['train_loss'], label='Train loss')
ax.plot(history_df['epoch'], history_df['val_loss'], label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('ISL Static Prototype — Loss')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation accuracy')
ax.plot(history_df['epoch'], history_df['val_macro_f1'], label='Validation macro F1')
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('ISL Static Prototype — Validation Metrics')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
assert BEST_MODEL_PATH.exists()
assert LAST_MODEL_PATH.exists()
assert np.isfinite(final_accuracy)
assert np.isfinite(final_macro_f1)
assert final_cm.shape == (NUM_CLASSES, NUM_CLASSES)
print()
print(' Best model saved')
print(' Last model saved')
print(' Validation metrics computed')
print(' Confusion matrix computed')
