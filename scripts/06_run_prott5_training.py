"""
ProtT5 Training Script for GPU Cluster
Converted from Jupyter notebook for batch execution
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Detect if we're running inside Singularity container
# Inside container, paths are mounted to /mnt
if os.path.exists('/mnt/code'):
    BASE_DIR = '/mnt'
    print("Running inside Singularity container")
else:
    BASE_DIR = '..'
    print("Running on local system")

# Set paths based on environment
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("\n" + "="*80)
print("PROTT5 TRAINING PIPELINE")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print("="*80 + "\n")


# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading datasets...")
# Load Veltri dataset
veltri_train = pd.read_csv(os.path.join(DATA_DIR, 'veltri_train.csv'))
veltri_val = pd.read_csv(os.path.join(DATA_DIR, 'veltri_val.csv'))
veltri_test = pd.read_csv(os.path.join(DATA_DIR, 'veltri_test.csv'))

# Load LMPred dataset
lmpred_train = pd.read_csv(os.path.join(DATA_DIR, 'lmpred_train.csv'))
lmpred_val = pd.read_csv(os.path.join(DATA_DIR, 'lmpred_val.csv'))
lmpred_test = pd.read_csv(os.path.join(DATA_DIR, 'lmpred_test.csv'))

print("=== Dataset Sizes ===")
print(f"Veltri - Train: {len(veltri_train)}, Val: {len(veltri_val)}, Test: {len(veltri_test)}")
print(f"LMPred - Train: {len(lmpred_train)}, Val: {len(lmpred_val)}, Test: {len(lmpred_test)}")
print()


# ============================================================================
# FUNCTIONS
# ============================================================================

def prepare_sequence_for_prott5(sequence):
    """
    ProtT5 requires spaces between amino acids
    AND replaces rare amino acids with X
    """
    # Replace rare amino acids (U, Z, O, B) with X
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    
    # Add spaces between amino acids
    spaced_seq = ' '.join(list(sequence))
    
    # Add ProtT5 prefix token
    formatted_seq = f"<AA2fold> {spaced_seq}"
    
    return formatted_seq


class ProtT5Dataset(Dataset):
    """PyTorch Dataset for ProtT5"""
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Format for ProtT5
        formatted_seq = prepare_sequence_for_prott5(sequence)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ProtT5Classifier(nn.Module):
    """ProtT5 + Classification Head"""
    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", n_classes=2, dropout=0.3, freeze_t5=False):
        super(ProtT5Classifier, self).__init__()
        
        print(f"Loading ProtT5 model: {model_name}")
        print("Note: This is a large model (~11GB), may take a few minutes to download...")
        
        # Use encoder-only version
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        
        # Get hidden size
        hidden_size = self.t5.config.d_model
        print(f"Hidden size: {hidden_size}")
        
        # Freeze T5 parameters if specified
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
            print("T5 parameters frozen - only training classifier")
        
        # Lightweight classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Get T5 embeddings
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Mean pooling (weighted by attention mask)
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask  # [batch_size, hidden_size]
        
        # Dropout + classify
        x = self.dropout(mean_embedding)
        logits = self.classifier(x)
        
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Clear cache periodically
        if len(all_preds) % 5 == 0:
            torch.cuda.empty_cache()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    return metrics


def train_prott5_model(train_df, val_df, test_df, dataset_name,
                       model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
                       batch_size=2, epochs=5, learning_rate=2e-5,
                       freeze_t5=False, use_scheduler=True):
    """Complete training pipeline for ProtT5"""
    print(f"\n{'='*60}")
    print(f"Training ProtT5 Model: {dataset_name}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze T5: {freeze_t5}")
    print(f"Use scheduler: {use_scheduler}")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Load tokenizer
    print("\nLoading ProtT5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    # Create datasets
    print("Creating datasets...")
    train_dataset = ProtT5Dataset(train_df, tokenizer, max_length=256)
    val_dataset = ProtT5Dataset(val_df, tokenizer, max_length=256)
    test_dataset = ProtT5Dataset(test_df, tokenizer, max_length=256)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("\nInitializing ProtT5 model...")
    model = ProtT5Classifier(model_name=model_name, n_classes=2, dropout=0.3, freeze_t5=freeze_t5)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (optional)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    # Early stopping
    best_val_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        # Update learning rate (if scheduler exists)
        if scheduler is not None:
            scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.2e}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n⚠ Early stopping! No improvement for {patience} epochs")
                break

        # Clear cache after each epoch
        torch.cuda.empty_cache()

    # Load best model
    print("\nLoading best model for testing...")
    model.load_state_dict(best_model_state)

    # Test
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    for metric, value in test_metrics.items():
        print(f"{metric.capitalize():15s}: {value:.4f}")

    return model, history, test_metrics


def plot_training_history(history, dataset_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{dataset_name} - Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy & F1
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].plot(epochs, history['val_f1'], 'g-', label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'{dataset_name} - Metrics')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'prott5_{dataset_name.lower()}_training.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training plot to: {save_path}")
    plt.close()


# ============================================================================
# MAIN TRAINING EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Train on Veltri dataset
    print("\n" + "="*80)
    print("TRAINING PROTT5 ON VELTRI DATASET")
    print("="*80 + "\n")
    
    veltri_model, veltri_history, veltri_results = train_prott5_model(
        veltri_train,
        veltri_val,
        veltri_test,
        dataset_name="Veltri",
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        batch_size=4,  # Adjust based on your GPU memory
        epochs=10,
        learning_rate=2e-5,
        freeze_t5=True,  # Freeze for faster training
        use_scheduler=False
    )
    
    plot_training_history(veltri_history, "Veltri")
    
    # Clear memory before next dataset
    del veltri_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "="*80)
    print("TRAINING PROTT5 ON LMPRED DATASET")
    print("="*80 + "\n")
    
    lmpred_model, lmpred_history, lmpred_results = train_prott5_model(
        lmpred_train,
        lmpred_val,
        lmpred_test,
        dataset_name="LMPred",
        model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        batch_size=2,           # ← REDUCE (unfrozen needs more memory)
        epochs=20,              # ← More epochs
        learning_rate=1e-5,     # ← LOWER when unfrozen
        freeze_t5=False,        # ← UNFREEZE for fine-tuning
        use_scheduler=True
    )
    
    plot_training_history(lmpred_history, "LMPred")
    
    # Save final results
    print("\n" + "="*80)
    print("SAVING FINAL RESULTS")
    print("="*80 + "\n")
    
    final_results = pd.DataFrame([
        {'Model': 'ProtT5', 'Dataset': 'Veltri', **veltri_results},
        {'Model': 'ProtT5', 'Dataset': 'LMPred', **lmpred_results}
    ])
    
    final_results.to_csv(os.path.join(RESULTS_DIR, 'results_final_all_models.csv'), index=False)
    print(f"Saved results to: {os.path.join(RESULTS_DIR, 'results_final_all_models.csv')}")
    
    # Plot comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    
    for idx, metric in enumerate(metrics_to_plot):
        veltri_val = veltri_results[metric]
        lmpred_val = lmpred_results[metric]
        
        axes[idx].bar(['Veltri', 'LMPred'], [veltri_val, lmpred_val], color=['#3498db', '#e74c3c'])
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_title(f'{metric.capitalize()}')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate([veltri_val, lmpred_val]):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle('ProtT5 Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    comparison_path = os.path.join(RESULTS_DIR, 'final_all_models_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {comparison_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("Results saved to:")
    print(f"  - {os.path.join(RESULTS_DIR, 'prott5_veltri_training.png')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'prott5_lmpred_training.png')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'results_final_all_models.csv')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'final_all_models_comparison.png')}")
