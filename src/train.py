import os
import sys
import tempfile

# CRITICAL: Set MPLCONFIGDIR before importing matplotlib to prevent permission errors
mpl_cache_dir = os.path.join(tempfile.gettempdir(), 'mpl_cache')
os.makedirs(mpl_cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache_dir

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless non-GUI backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessor import TitanicPreprocessor
from src.dataset import create_data_loaders, TitanicTorchDataset
from src.model import TitanicDeepNet

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * len(X_batch)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * len(X_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

    total = len(all_targets)
    epoch_loss = running_loss / total
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        'loss': epoch_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'targets': all_targets,
        'preds': all_preds,
        'probs': all_probs
    }

def run_cross_validation(X, y, n_splits=5):
    """Run 5-Fold Stratified Cross Validation and compare with baselines."""
    print(f"\n--- Running {n_splits}-Fold Stratified Cross Validation ---", flush=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    nn_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    lr_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    rf_metrics = {'accuracy': [], 'f1': [], 'auc': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. Deep Learning Model
        train_loader, val_loader = create_data_loaders(X_tr, y_tr, X_va, y_va, batch_size=32)
        model = TitanicDeepNet(input_dim=X.shape[1], hidden_dim=64, dropout_rate=0.2)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_auc = 0.0

        for epoch in range(40):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_res = evaluate(model, val_loader, criterion)
            if val_res['loss'] < best_val_loss:
                best_val_loss = val_res['loss']
                best_val_acc = val_res['accuracy']
                best_val_f1 = val_res['f1']
                best_val_auc = val_res['auc']

        nn_metrics['accuracy'].append(best_val_acc)
        nn_metrics['f1'].append(best_val_f1)
        nn_metrics['auc'].append(best_val_auc)

        # 2. Logistic Regression Baseline
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_tr, y_tr)
        lr_pred = lr.predict(X_va)
        lr_prob = lr.predict_proba(X_va)[:, 1]
        lr_metrics['accuracy'].append(accuracy_score(y_va, lr_pred))
        lr_metrics['f1'].append(f1_score(y_va, lr_pred, zero_division=0))
        lr_metrics['auc'].append(roc_auc_score(y_va, lr_prob))

        # 3. Random Forest Baseline
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_va)
        rf_prob = rf.predict_proba(X_va)[:, 1]
        rf_metrics['accuracy'].append(accuracy_score(y_va, rf_pred))
        rf_metrics['f1'].append(f1_score(y_va, rf_pred, zero_division=0))
        rf_metrics['auc'].append(roc_auc_score(y_va, rf_prob))

        print(f"  Fold {fold}/{n_splits} Complete | Deep Net Acc: {best_val_acc:.4f} | LR Acc: {lr_metrics['accuracy'][-1]:.4f} | RF Acc: {rf_metrics['accuracy'][-1]:.4f}", flush=True)

    summary = {
        'Deep Neural Network': {
            'Accuracy': float(np.mean(nn_metrics['accuracy'])),
            'F1-Score': float(np.mean(nn_metrics['f1'])),
            'ROC-AUC': float(np.mean(nn_metrics['auc']))
        },
        'Logistic Regression': {
            'Accuracy': float(np.mean(lr_metrics['accuracy'])),
            'F1-Score': float(np.mean(lr_metrics['f1'])),
            'ROC-AUC': float(np.mean(lr_metrics['auc']))
        },
        'Random Forest': {
            'Accuracy': float(np.mean(rf_metrics['accuracy'])),
            'F1-Score': float(np.mean(rf_metrics['f1'])),
            'ROC-AUC': float(np.mean(rf_metrics['auc']))
        }
    }

    print("\n=== Model Benchmark Comparison (5-Fold CV) ===", flush=True)
    for model_name, scores in summary.items():
        print(f"[{model_name}] Accuracy: {scores['Accuracy']:.4f} | F1: {scores['F1-Score']:.4f} | AUC: {scores['ROC-AUC']:.4f}", flush=True)

    return summary

def compute_permutation_importance(model, X_val, y_val, feature_names):
    """Compute permutation importance for deep neural network features."""
    baseline_acc = accuracy_score(y_val, model.predict(X_val))
    importances = {}

    for i, col in enumerate(feature_names):
        X_perm = X_val.copy()
        np.random.seed(42)
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_acc = accuracy_score(y_val, model.predict(X_perm))
        importance = max(0.0, baseline_acc - perm_acc)
        importances[col] = float(importance)

    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_imp

def train_and_export():
    """Main training, evaluation, plot generation, and artifact export pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'titanic.csv')
    saved_models_dir = os.path.join(project_root, 'saved_models')
    plots_dir = os.path.join(project_root, 'app', 'static', 'plots')

    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}", flush=True)
    raw_df = pd.read_csv(data_path)
    print(f"Loaded {len(raw_df)} passengers.", flush=True)

    # Fit preprocessor
    preprocessor = TitanicPreprocessor()
    X = preprocessor.fit_transform(raw_df)
    y = raw_df['Survived'].values.astype(np.float32)

    print(f"Preprocessed features shape: {X.shape}", flush=True)
    print(f"Feature columns ({len(preprocessor.feature_names)}): {preprocessor.feature_names}", flush=True)

    # Benchmark comparison via Cross Validation
    cv_benchmark = run_cross_validation(X, y, n_splits=5)

    # Train / Validation Split for final model and learning curve plots
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32)

    model = TitanicDeepNet(input_dim=X.shape[1], hidden_dim=64, dropout_rate=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 100
    patience = 20
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_state = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print("\n--- Training Final Deep Neural Network ---", flush=True)
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_res = evaluate(model, val_loader, criterion)

        scheduler.step(val_res['loss'])

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_res['loss'])
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_res['accuracy'])

        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']:.4f}", flush=True)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Restoring best model weights.", flush=True)
            break

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on validation set
    final_val_res = evaluate(model, val_loader, criterion)
    print("\n=== Final Deep Learning Model Evaluation ===", flush=True)
    print(f"Validation Loss:      {final_val_res['loss']:.4f}", flush=True)
    print(f"Validation Accuracy:  {final_val_res['accuracy'] * 100:.2f}%", flush=True)
    print(f"Validation Precision: {final_val_res['precision'] * 100:.2f}%", flush=True)
    print(f"Validation Recall:    {final_val_res['recall'] * 100:.2f}%", flush=True)
    print(f"Validation F1-Score:  {final_val_res['f1'] * 100:.2f}%", flush=True)
    print(f"Validation ROC-AUC:   {final_val_res['auc']:.4f}", flush=True)

    # Feature Importance
    sorted_importances = compute_permutation_importance(
        model, X_val, y_val, preprocessor.feature_names
    )

    # 1. Plot Loss Curve
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='#dc2626', linewidth=2, linestyle='--')
    plt.title('Deep Learning Training & Validation Loss', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('BCE Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    loss_curve_path = os.path.join(plots_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()

    # 2. Plot Accuracy Curve
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(history['train_acc'], label='Train Accuracy', color='#16a34a', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#9333ea', linewidth=2, linestyle='--')
    plt.title('Deep Learning Training & Validation Accuracy', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    acc_curve_path = os.path.join(plots_dir, 'accuracy_curve.png')
    plt.savefig(acc_curve_path)
    plt.close()

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(final_val_res['targets'], final_val_res['preds'])
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Did Not Survive (0)', 'Survived (1)'], fontsize=9)
    plt.yticks(tick_marks, ['Did Not Survive (0)', 'Survived (1)'], fontsize=9)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold', fontsize=12)
    plt.ylabel('True Passenger Outcome', fontsize=10)
    plt.xlabel('Predicted Outcome', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # 4. Plot ROC Curve
    fpr, tpr, _ = roc_curve(final_val_res['targets'], final_val_res['probs'])
    plt.figure(figsize=(6, 4.5), dpi=150)
    plt.plot(fpr, tpr, color='#0284c7', lw=2.5, label=f"ROC Curve (AUC = {final_val_res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    # 5. Plot Top Feature Importances
    top_k = min(10, len(sorted_importances))
    top_features = [x[0] for x in sorted_importances[:top_k]][::-1]
    top_vals = [x[1] for x in sorted_importances[:top_k]][::-1]

    plt.figure(figsize=(7, 5), dpi=150)
    plt.barh(range(top_k), top_vals, color='#4f46e5', align='center')
    plt.yticks(range(top_k), top_features, fontsize=9)
    plt.xlabel('Permutation Accuracy Drop', fontsize=10)
    plt.title('Deep Neural Network: Top Feature Importances', fontsize=13, fontweight='bold', pad=10)
    plt.grid(True, axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    feat_imp_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(feat_imp_path)
    plt.close()

    print(f"Generated and saved all evaluation plots to: {plots_dir}", flush=True)

    # Save artifacts
    model_path = os.path.join(saved_models_dir, 'titanic_deep_model.pt')
    preprocessor_path = os.path.join(saved_models_dir, 'preprocessor.pkl')
    metrics_path = os.path.join(saved_models_dir, 'metrics.json')

    # Save full model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X.shape[1],
        'hidden_dim': 64,
        'dropout_rate': 0.2,
        'feature_names': preprocessor.feature_names
    }, model_path)
    print(f"Saved PyTorch deep learning model to: {model_path}", flush=True)

    # Save fitted preprocessor
    preprocessor.save(preprocessor_path)
    print(f"Saved fitted preprocessor to: {preprocessor_path}", flush=True)

    # Export metrics JSON
    metrics_data = {
        'total_passengers': len(raw_df),
        'input_features_count': int(X.shape[1]),
        'feature_names': preprocessor.feature_names,
        'validation_metrics': {
            'loss': round(float(final_val_res['loss']), 4),
            'accuracy': round(float(final_val_res['accuracy']), 4),
            'precision': round(float(final_val_res['precision']), 4),
            'recall': round(float(final_val_res['recall']), 4),
            'f1_score': round(float(final_val_res['f1']), 4),
            'roc_auc': round(float(final_val_res['auc']), 4),
            'confusion_matrix': cm.tolist()
        },
        'benchmark_comparison': cv_benchmark,
        'top_feature_importances': dict(sorted_importances[:10])
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Exported metrics report to: {metrics_path}", flush=True)

    return metrics_data

if __name__ == '__main__':
    train_and_export()
