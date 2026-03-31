import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score
)
from transformers import (
    BertTokenizer, BertModel,
    AutoTokenizer, EsmModel,
    T5Tokenizer, T5EncoderModel
)
import warnings
import os
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

PROTBERT_PTH = '../results/protbert_veltri_final.pth'
ESM2_PTH = '../results/esm2_veltri_model.pth'
PROTT5_PTH = '../results/prott5_veltri_final.pth'

VELTRI_TEST = '../data/veltri_test.csv'
LMPRED_TEST = '../data/lmpred_test.csv'

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/error_analysis'
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0,
}
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0,
}


def seq_hydrophobicity(seq):
    vals = [HYDROPHOBICITY.get(aa, 0.0) for aa in seq.upper()]
    return np.mean(vals) if vals else 0.0


def seq_charge(seq):
    return sum(CHARGE.get(aa, 0) for aa in seq.upper())


def seq_length(seq):
    return len(seq)


class ProtBERTClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ESM2Classifier(nn.Module):
    def __init__(self, model_name='facebook/esm2_t6_8M_UR50D',
                 n_classes=2, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


class ProtT5Classifier(nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_half_uniref50-enc',
                 n_classes=2, dropout=0.3, freeze_t5=False):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.t5.config.d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return self.classifier(self.dropout(mean_emb))


def load_model(model_obj, pth_path, label):
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'Weigths: {pth_path}')
    else:
        print(f'WARNING: {pth_path} does not exist - pretrained weights')
    model_obj.eval().to(device)
    print(f'{label} is ready.')
    return model_obj


def format_sequence(sequence, model_type):
    if model_type == 'bert':
        return ' '.join(list(sequence))
    elif model_type == 't5':
        seq = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        return f"<AA2fold> {' '.join(list(seq))}"
    else:
        return sequence


def run_inference(df, model, tokenizer, model_type):
    all_preds = []
    all_probs = []
    sequences = df['sequence'].tolist()
    n = len(sequences)

    for start in range(0, n, BATCH_SIZE):
        batch = sequences[start: start + BATCH_SIZE]
        formatted = [format_sequence(s, model_type) for s in batch]

        encoding = tokenizer(
            formatted,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(logits, dim=1)

        all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())  # P(AMP)

        if (start // BATCH_SIZE + 1) % 5 == 0:
            print(f'{min(start + BATCH_SIZE, n)}/{n} processed...')

    return np.array(all_preds), np.array(all_probs)


def build_error_df(df, true_labels, pred_labels, probs, model_name, dataset_name):
    """
    error_type: TP, TN, FP, FN
    """
    records = []
    for i, (seq, true, pred, prob) in enumerate(
            zip(df['sequence'], true_labels, pred_labels, probs)):

        if true == 1 and pred == 1:
            error_type = 'TP'
        elif true == 0 and pred == 0:
            error_type = 'TN'
        elif true == 0 and pred == 1:
            error_type = 'FP'  # non-AMP predicted as AMP
        else:
            error_type = 'FN'  # AMP missed - most important error

        records.append({
            'model': model_name,
            'dataset': dataset_name,
            'sequence': seq,
            'length': seq_length(seq),
            'charge': seq_charge(seq),
            'hydrophobicity': seq_hydrophobicity(seq),
            'true_label': true,
            'pred_label': pred,
            'prob_amp': round(prob, 4),
            'error_type': error_type,
            'is_error': error_type in ('FP', 'FN'),
        })

    return pd.DataFrame(records)


def plot_confusion_matrix(true_labels, pred_labels, model_name,
                          dataset_name, output_dir):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['non-AMP', 'AMP'],
                yticklabels=['non-AMP', 'AMP'],
                linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'{model_name} - {dataset_name}\nConfusion Matrix',
                 fontsize=12, fontweight='bold')

    # FN/FP annotations
    ax.text(0.5, 1.5, 'FN\n(missed AMP)',
            ha='center', va='center', fontsize=8, color='#e74c3c',
            fontweight='bold')
    ax.text(1.5, 0.5, 'FP\n(false alarm)',
            ha='center', va='center', fontsize=8, color='#e67e22',
            fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'cm_{model_name.lower().replace("-", "_")}_{dataset_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_error_profiles(error_df, model_name, dataset_name, output_dir):
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e67e22', 'FN': '#e74c3c'}
    props = [
        ('length', 'Sequence length (AA)'),
        ('charge', 'Total charge'),
        ('hydrophobicity', 'Average hydrophobicity (KD)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'{model_name} - {dataset_name}\nError profile by physicochemical properties',
        fontsize=12, fontweight='bold'
    )

    for ax, (col, xlabel) in zip(axes, props):
        for etype, color in colors.items():
            subset = error_df[error_df['error_type'] == etype][col]
            if len(subset) == 0:
                continue
            ax.hist(subset, bins=20, alpha=0.55, color=color,
                    label=f'{etype} (n={len(subset)})', density=True)
            ax.axvline(subset.mean(), color=color, linewidth=2,
                       linestyle='--', alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(xlabel, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'error_profiles_{model_name.lower().replace("-", "_")}_{dataset_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_confidence_errors(error_df, model_name, dataset_name, output_dir):
    colors = {'TP': '#2ecc71', 'TN': '#3498db', 'FP': '#e67e22', 'FN': '#e74c3c'}
    fig, ax = plt.subplots(figsize=(9, 5))

    for etype, color in colors.items():
        sub = error_df[error_df['error_type'] == etype]
        if len(sub) == 0:
            continue
        ax.scatter(sub['length'], sub['prob_amp'],
                   c=color, alpha=0.5, s=20,
                   label=f'{etype} (n={len(sub)})')

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Decision boundary (0.5)')
    ax.set_xlabel('Sequence length (AA)', fontsize=11)
    ax.set_ylabel('P(AMP) - model confidence', fontsize=11)
    ax.set_title(f'{model_name} - {dataset_name}\nConfidence vs length',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='center right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'confidence_{model_name.lower().replace("-", "_")}_{dataset_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_three_model_fn_comparison(all_error_dfs, dataset_name, output_dir):
    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    props = [
        ('length', 'Length (AA)'),
        ('charge', 'Charge'),
        ('hydrophobicity', 'Hydrophobicity'),
    ]
    colors = {'ESM-2': '#3498db', 'ProtBERT': '#e74c3c', 'ProtT5': '#2ecc71'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'False Negatives (missed AMPs) - comparison of three models\n{dataset_name}',
        fontsize=12, fontweight='bold'
    )

    for ax, (col, xlabel) in zip(axes, props):
        for mname in models:
            df_m = all_error_dfs.get((mname, dataset_name))
            if df_m is None:
                continue
            fn_vals = df_m[df_m['error_type'] == 'FN'][col]
            if len(fn_vals) == 0:
                continue
            ax.hist(fn_vals, bins=15, alpha=0.55,
                    color=colors[mname],
                    label=f'{mname} (n={len(fn_vals)})',
                    density=True)
            ax.axvline(fn_vals.mean(), color=colors[mname],
                       linewidth=2, linestyle='--', alpha=0.9)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(xlabel, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'fn_comparison_3models_{dataset_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_error_rate_summary(summary_rows, output_dir):
    """
    FN rate = FN / (FN + TP)
    FP rate = FP / (FP + TN)
    """
    df = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('FN rate and FP rate per model and dataset',
                 fontsize=13, fontweight='bold')

    for ax, (metric, title, color) in zip(axes, [
        ('fn_rate', 'FN rate - missed AMPs (%)', '#e74c3c'),
        ('fp_rate', 'FP rate - false alarms (%)', '#e67e22'),
    ]):
        pivot = df.pivot(index='model', columns='dataset', values=metric) * 100
        pivot.plot(kind='bar', ax=ax, color=['#3498db', '#9b59b6'],
                   alpha=0.8, edgecolor='white', width=0.6)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('%', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        ax.legend(title='Dataset', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'error_rate_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


if __name__ == '__main__':
    print('ERROR ANALYSIS - ProtBERT · ESM-2 · ProtT5')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = {}
    for name, path in [('Veltri', VELTRI_TEST), ('LMPred', LMPRED_TEST)]:
        df = pd.read_csv(path)
        print(f'\n{name}: {len(df)} sequences'
              f'(AMP={df["label"].sum()}, non-AMP={(df["label"] == 0).sum()})')
        datasets[name] = df

    all_error_dfs = {}  # {(model_name, dataset_name): error_df}
    summary_rows = []
    all_csv_rows = []

    model_configs = [
        ('ESM-2', lambda: ESM2Classifier(),
         lambda: AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'),
         ESM2_PTH, 'esm'),
        ('ProtBERT', lambda: ProtBERTClassifier(),
         lambda: BertTokenizer.from_pretrained('Rostlab/prot_bert',
                                               do_lower_case=False),
         PROTBERT_PTH, 'bert'),
        ('ProtT5', lambda: ProtT5Classifier(freeze_t5=True),
         lambda: T5Tokenizer.from_pretrained(
             'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False),
         PROTT5_PTH, 't5'),
    ]

    for model_name, model_fn, tok_fn, pth, mtype in model_configs:
        print(f'── {model_name}')

        tokenizer = tok_fn()
        model = load_model(model_fn(), pth, model_name)

        for dataset_name, df in datasets.items():
            print(f'\n  [{dataset_name}]')
            true_labels = df['label'].values

            preds, probs = run_inference(df, model, tokenizer, mtype)

            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average='binary')
            auc = roc_auc_score(true_labels, probs)
            cm = confusion_matrix(true_labels, preds)
            tn, fp, fn, tp = cm.ravel()
            fn_rate = fn / (fn + tp + 1e-9)
            fp_rate = fp / (fp + tn + 1e-9)

            print(f'Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}')
            print(f'TP={tp}  TN={tn}  FP={fp}  FN={fn}')
            print(f'FN rate={fn_rate:.3f}  FP rate={fp_rate:.3f}')

            plot_confusion_matrix(true_labels, preds,
                                  model_name, dataset_name, OUTPUT_DIR)

            error_df = build_error_df(df, true_labels, preds, probs,
                                      model_name, dataset_name)
            all_error_dfs[(model_name, dataset_name)] = error_df
            all_csv_rows.append(error_df)

            plot_error_profiles(error_df, model_name, dataset_name, OUTPUT_DIR)
            plot_confidence_errors(error_df, model_name, dataset_name, OUTPUT_DIR)

            summary_rows.append({
                'model': model_name,
                'dataset': dataset_name,
                'accuracy': round(acc, 4),
                'f1': round(f1, 4),
                'auc': round(auc, 4),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'fn_rate': round(fn_rate, 4),
                'fp_rate': round(fp_rate, 4),
            })

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print('\n── COMPARISON - all three models ──────────────────────────────────────')
    for dataset_name in datasets:
        plot_three_model_fn_comparison(all_error_dfs, dataset_name, OUTPUT_DIR)

    plot_error_rate_summary(summary_rows, OUTPUT_DIR)

    full_error_df = pd.concat(all_csv_rows, ignore_index=True)
    errors_only = full_error_df[full_error_df['is_error']]
    csv_path = os.path.join(OUTPUT_DIR, 'error_analysis_summary.csv')
    errors_only.to_csv(csv_path, index=False)
    print(f'\nTotal misclassified samples saved: {len(errors_only)}')
    print(f'Saved at: {csv_path}')

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)

    print('DONE!')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
