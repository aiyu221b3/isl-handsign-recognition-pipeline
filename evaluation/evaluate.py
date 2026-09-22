from isl_static_prototype.data_pipeline import PROJECT_ROOT
from isl_static_prototype.vision.graph_model import ISLGraphClassifier
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
DEVICE = torch.device('cpu')
MODEL_PATH = PROJECT_ROOT / 'models' / 'isl_static_best.pt'
TEST_FEATURE_PATH = PROJECT_ROOT / 'data' / 'features' / 'test.npz'
ARTIFACT_ROOT = PROJECT_ROOT / 'artifacts'
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ['Hello', 'IloveYou', 'No', 'Please', 'Thanks', 'Yes']
NUM_CLASSES = len(CLASS_NAMES)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Best model not found:\n{MODEL_PATH}')
if not TEST_FEATURE_PATH.exists():
    raise FileNotFoundError(f'Test feature dataset not found:\n{TEST_FEATURE_PATH}')
test_data = np.load(TEST_FEATURE_PATH, allow_pickle=False)
X_node_test = torch.from_numpy(test_data['X_node']).float()
X_global_test = torch.from_numpy(test_data['X_global']).float()
y_test = test_data['y'].astype(np.int64)
test_filenames = test_data['filenames']
print('=' * 70)
print('FINAL TEST DATA')
print('=' * 70)
print(f'Samples:          {len(y_test)}')
print(f'Node features:    {X_node_test.shape}')
print(f'Global features:  {X_global_test.shape}')
print('Classes:')
for class_id, class_name in enumerate(CLASS_NAMES):
    count = int(np.sum(y_test == class_id))
    print(f' {class_id}: {class_name:10s} {count}')
model = ISLGraphClassifier(node_features=5, global_features=35, graph_hidden=48, classifier_hidden=64, num_classes=NUM_CLASSES, dropout=0.1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
best_epoch = checkpoint.get('epoch', None)
best_validation_f1 = checkpoint.get('best_val_macro_f1', None)
print()
print('Loaded checkpoint:')
print(f' Epoch:              {best_epoch}')
print(f' Best validation F1: {best_validation_f1}')
with torch.no_grad():
    logits = model(X_node_test, X_global_test)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
predictions_np = predictions.cpu().numpy()
probabilities_np = probabilities.cpu().numpy()
test_accuracy = accuracy_score(y_test, predictions_np)
test_macro_f1 = f1_score(y_test, predictions_np, labels=np.arange(NUM_CLASSES), average='macro', zero_division=0)
test_precision = precision_score(y_test, predictions_np, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
test_recall = recall_score(y_test, predictions_np, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
test_f1_per_class = f1_score(y_test, predictions_np, labels=np.arange(NUM_CLASSES), average=None, zero_division=0)
test_cm = confusion_matrix(y_test, predictions_np, labels=np.arange(NUM_CLASSES))
print()
print('=' * 70)
print('FINAL TEST RESULTS')
print('=' * 70)
print(f'Accuracy:       {test_accuracy:.4f}')
print(f'Macro F1:       {test_macro_f1:.4f}')
print(f'Correct:        {int(np.sum(predictions_np == y_test))} / {len(y_test)}')
print(f'Incorrect:      {int(np.sum(predictions_np != y_test))} / {len(y_test)}')
print()
print('=' * 70)
print('PER-CLASS TEST METRICS')
print('=' * 70)
for class_id, class_name in enumerate(CLASS_NAMES):
    print(f'{class_name:10s} | precision={test_precision[class_id]:.4f} | recall={test_recall[class_id]:.4f} | F1={test_f1_per_class[class_id]:.4f}')
print()
print('=' * 70)
print('TEST CONFUSION MATRIX')
print('=' * 70)
cm_df = pd.DataFrame(test_cm, index=CLASS_NAMES, columns=CLASS_NAMES)
display(cm_df)
prediction_records = []
for index in range(len(y_test)):
    true_id = int(y_test[index])
    predicted_id = int(predictions_np[index])
    prediction_records.append({'filename': str(test_filenames[index]), 'true_class': CLASS_NAMES[true_id], 'predicted_class': CLASS_NAMES[predicted_id], 'correct': bool(true_id == predicted_id), 'confidence': float(probabilities_np[index, predicted_id])})
predictions_df = pd.DataFrame(prediction_records)
predictions_path = ARTIFACT_ROOT / 'test_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)
cm_path = ARTIFACT_ROOT / 'test_confusion_matrix.csv'
cm_df.to_csv(cm_path)
benchmark_node = X_node_test[:1].to(DEVICE)
benchmark_global = X_global_test[:1].to(DEVICE)
with torch.no_grad():
    for _ in range(20):
        _ = model(benchmark_node, benchmark_global)
iterations = 1000
start = time.perf_counter()
with torch.no_grad():
    for _ in range(iterations):
        _ = model(benchmark_node, benchmark_global)
elapsed = time.perf_counter() - start
model_latency_ms = elapsed / iterations * 1000
model_fps = 1000.0 / model_latency_ms
print()
print('=' * 70)
print('MODEL-ONLY CPU INFERENCE')
print('=' * 70)
print(f'Latency: {model_latency_ms:.4f} ms/image')
print(f'Throughput: {model_fps:.1f} FPS')
final_metrics = {'evaluation_type': 'held_out_test', 'test_samples': int(len(y_test)), 'correct': int(np.sum(predictions_np == y_test)), 'incorrect': int(np.sum(predictions_np != y_test)), 'accuracy': float(test_accuracy), 'macro_f1': float(test_macro_f1), 'per_class': {CLASS_NAMES[class_id]: {'precision': float(test_precision[class_id]), 'recall': float(test_recall[class_id]), 'f1': float(test_f1_per_class[class_id])} for class_id in range(NUM_CLASSES)}, 'model_checkpoint': str(MODEL_PATH), 'best_validation_epoch': int(best_epoch) if best_epoch is not None else None, 'best_validation_macro_f1': float(best_validation_f1) if best_validation_f1 is not None else None, 'model_latency_ms': float(model_latency_ms), 'model_fps': float(model_fps)}
metrics_path = ARTIFACT_ROOT / 'final_test_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
mistakes_df = predictions_df[~predictions_df['correct']].copy()
print()
if len(mistakes_df) == 0:
    print('No test-set classification errors.')
else:
    print('=' * 70)
    print('TEST-SET MISTAKES')
    print('=' * 70)
    display(mistakes_df[['filename', 'true_class', 'predicted_class', 'confidence']])
print()
print('=' * 70)
print('FINAL EVALUATION COMPLETE')
print('=' * 70)
print(f'Predictions:       {predictions_path}')
print(f'Confusion matrix:  {cm_path}')
print(f'Metrics:           {metrics_path}')
print()
print('Held-out test evaluated once')
print('No test-set training')
print('No test-set model selection')
print('Per-class metrics saved')
print('Confusion matrix saved')
print('Inference latency measured')
