import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
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

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/physicochemical_corr'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_SEQUENCES = {
    'AMP_1 (magainin)': 'GIGKFLHSAKKFGKAFVGEIMNS',
    'AMP_2 (defensin)': 'ACYCRIPACIAGERRYGTCIYQGRLWAFCC',
    'nonAMP_1': 'MKLLFAIPVAVALAAGVQPQDAPSVAQKLEE',
    'nonAMP_2': 'GASVVDLNKLTQPDQSAGAKNLGKISQTLK',
}

SPECIAL_TOKENS = {
    '[CLS]', '[SEP]', '<cls>', '<eos>', '<pad>',
    '<unk>', '</s>', '<s>', '▁'
}

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0,
}

# +1 = positive (K, R), -1 = negative (D, E), 0 = neutral
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0,
}

# Molecular weights (Dalton)
MOLECULAR_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
    'X': 110.0,
}

PROPERTIES = {
    'Hydrophobicity': HYDROPHOBICITY,
    'Charge': CHARGE,
    'Molecular weight': MOLECULAR_WEIGHT,
}

class ProtBERTClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            'Rostlab/prot_bert', output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        return logits, outputs.attentions


class ESM2Classifier(nn.Module):
    def __init__(self, model_name='facebook/esm2_t6_8M_UR50D',
                 n_classes=2, dropout=0.3):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        return logits, outputs.attentions


class ProtT5Classifier(nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_half_uniref50-enc',
                 n_classes=2, dropout=0.3, freeze_t5=False):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name, output_attentions=True)
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
        logits = self.classifier(self.dropout(mean_emb))
        return logits, outputs.attentions


def load_model(model_obj, pth_path, label):
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'Weights: {pth_path}')
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


def safe_name(s):
    return s[:15].replace(' ', '_').replace('(', '').replace(')', '')

def get_attention_and_tokens(sequence, model, tokenizer, model_type):

    formatted = format_sequence(sequence, model_type)

    encoding = tokenizer(
        formatted,
        return_tensors='pt',
        add_special_tokens=True,
        max_length=256,
        truncation=True
    ).to(device)

    with torch.no_grad():
        logits, attentions = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )

    pred_label = 'AMP' if torch.argmax(logits, dim=1).item() == 1 else 'non-AMP'

    # Stack: (layers, heads, L, L) → average over layers and heads
    attn_stack = torch.stack(attentions, dim=0).squeeze(1).cpu().numpy()
    mean_attn = attn_stack.mean(axis=0).mean(axis=0)  # (L, L)

    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    keep = [i for i, t in enumerate(tokens)
            if t not in SPECIAL_TOKENS and len(t.strip()) > 0]
    clean_tokens = [tokens[i].strip().upper().replace('▁', '') for i in keep]

    importance = mean_attn[np.ix_(keep, keep)].sum(axis=0)
    importance = importance / (importance.sum() + 1e-9)

    return clean_tokens, importance, pred_label


def get_physicochemical_profile(tokens, prop_dict):
    return np.array([prop_dict.get(aa, 0.0) for aa in tokens])


def plot_single_sequence(seq_name, sequence, model_name,
                         tokens, importance, pred_label, output_dir):
    n = len(tokens)
    positions = np.arange(n)

    hydro = get_physicochemical_profile(tokens, HYDROPHOBICITY)
    charge = get_physicochemical_profile(tokens, CHARGE)
    mw = get_physicochemical_profile(tokens, MOLECULAR_WEIGHT)

    fig, axes = plt.subplots(3, 1, figsize=(max(10, n * 0.5), 13))
    fig.suptitle(
        f'{model_name}  |  {seq_name}  |  Prediction: {pred_label}\n{sequence}',
        fontsize=12, fontweight='bold', y=1.01
    )

    ax = axes[0]
    bar_colors = [
        '#e74c3c' if imp > np.percentile(importance, 75)
        else '#3498db' if imp < np.percentile(importance, 25)
        else '#95a5a6'
        for imp in importance
    ]
    ax.bar(positions, importance, color=bar_colors, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(tokens, fontsize=9)
    ax.set_ylabel('Attention importance', fontsize=10)
    ax.set_title('Attention per position (red = top 25%, blue = bottom 25%)',
                 fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for i, aa in enumerate(tokens):
        if aa in ('K', 'R'):
            ax.get_xticklabels()[i].set_color('#e74c3c')
            ax.get_xticklabels()[i].set_fontweight('bold')
        elif aa in ('D', 'E'):
            ax.get_xticklabels()[i].set_color('#3498db')

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    l1, = ax2.plot(positions, hydro, 'o-', color='#e67e22',
                   linewidth=2, markersize=5, label='Hydrophobicity (KD)')
    l2, = ax2.plot(positions, mw / mw.max(), 's--', color='#9b59b6',
                   linewidth=1.5, markersize=4, label='Molecular weight (norm.)', alpha=0.7)
    l3, = ax2_twin.bar(positions, charge,
                       color=['#e74c3c' if c > 0 else '#3498db' if c < 0 else '#bdc3c7'
                              for c in charge],
                       alpha=0.4, width=0.4, label='Charge').findobj()[0].__class__

    from matplotlib.patches import Patch
    handles = [l1, l2,
               Patch(facecolor='#e74c3c', alpha=0.5, label='Charge (+)'),
               Patch(facecolor='#3498db', alpha=0.5, label='Charge (-)')]
    ax2.legend(handles=handles, fontsize=8, loc='upper right')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(tokens, fontsize=9)
    ax2.set_ylabel('Hydrophobicity / Molecular weight', fontsize=9)
    ax2_twin.set_ylabel('Charge', fontsize=9)
    ax2.set_title('Physicochemical profile of the sequence', fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    ax3 = axes[2]

    props_to_plot = [
        (hydro, 'Hydrophobicity (KD)', '#e67e22'),
        (charge, 'Charge', '#e74c3c'),
        (mw / mw.max(), 'Molecular weight (norm.)', '#9b59b6'),
    ]

    for prop_vals, prop_name, color in props_to_plot:
        if len(np.unique(prop_vals)) < 2:
            continue
        r, p = pearsonr(importance, prop_vals)
        rho, _ = spearmanr(importance, prop_vals)
        ax3.scatter(prop_vals, importance, color=color, alpha=0.7,
                    s=60, label=f'{prop_name}  r={r:.2f}, ρ={rho:.2f}  '
                                f'({"*" if p < 0.05 else "ns"})')

    ax3.set_xlabel('Physicochemical value', fontsize=10)
    ax3.set_ylabel('Attention importance', fontsize=10)
    ax3.set_title('Correlation: attention vs physicochemical properties\n'
                  '(* = statistically significant, p<0.05)', fontsize=10)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'physicochem_{model_name.lower().replace("-", "_")}_{safe_name(seq_name)}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_three_model_comparison(seq_name, sequence, results, output_dir):

    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    props = [
        ('Hydrophobicity', HYDROPHOBICITY, '#e67e22'),
        ('Charge', CHARGE, '#e74c3c'),
        ('Molecular weight', MOLECULAR_WEIGHT, '#9b59b6'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f'Comparison: attention vs physicochemical  |  {seq_name}\n{sequence}',
        fontsize=13, fontweight='bold', y=1.01
    )

    for row, mname in enumerate(models):
        if mname not in results:
            continue
        tokens, importance, pred_label = results[mname]

        for col, (prop_name, prop_dict, color) in enumerate(props):
            ax = axes[row, col]
            prop_vals = get_physicochemical_profile(tokens, prop_dict)

            ax.scatter(prop_vals, importance, color=color, alpha=0.75, s=50)

            for i, (x, y, aa) in enumerate(zip(prop_vals, importance, tokens)):
                ax.annotate(aa, (x, y), fontsize=7, alpha=0.8,
                            xytext=(2, 2), textcoords='offset points')

            if len(np.unique(prop_vals)) > 1:
                r, p = pearsonr(importance, prop_vals)
                rho, _ = spearmanr(importance, prop_vals)
                sig = '*' if p < 0.05 else 'ns'
                ax.set_title(
                    f'{mname} - {prop_name}\nr={r:.2f}  ρ={rho:.2f}  ({sig})',
                    fontsize=9, fontweight='bold'
                )

                z = np.polyfit(prop_vals, importance, 1)
                xline = np.linspace(prop_vals.min(), prop_vals.max(), 50)
                ax.plot(xline, np.poly1d(z)(xline), '--',
                        color=color, alpha=0.5, linewidth=1)
            else:
                ax.set_title(f'{mname} - {prop_name}', fontsize=9)

            ax.set_xlabel(prop_name, fontsize=8)
            ax.set_ylabel('Attention importance' if col == 0 else '', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'physicochem_comparison_3models_{safe_name(seq_name)}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_summary_heatmap(summary_rows, output_dir):

    df = pd.DataFrame(summary_rows)
    pivot_cols = [c for c in df.columns if c not in ('Model', 'Sequence', 'Prediction')]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Pearson r - attention vs physicochemical (all models and sequences)',
                 fontsize=13, fontweight='bold')

    for ax, metric in zip(axes, ['r_hydro', 'r_charge']):
        if metric not in df.columns:
            continue
        pivot = df.pivot(index='Model', columns='Sequence', values=metric)
        sns.heatmap(pivot, ax=ax, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        title = 'Hydrophobicity' if metric == 'r_hydro' else 'Charge'
        ax.set_title(f'r vs {title}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'physicochem_summary_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


if __name__ == '__main__':
    print('PHYSICOCHEMICAL CORRELATION - ProtBERT · ESM-2 · ProtT5')
    print(f'Start : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
        print(f'VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # all_results[seq_name][model_name] = (tokens, importance, pred_label)
    all_results = {name: {} for name in TEST_SEQUENCES}
    summary_rows = []

    print('\n── ESM-2 ──────────────────────────────────────────────────────────')
    esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm2_mdl = load_model(ESM2Classifier(), ESM2_PTH, 'ESM-2')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        tokens, imp, pred = get_attention_and_tokens(seq, esm2_mdl, esm2_tok, 'esm')
        all_results[name]['ESM-2'] = (tokens, imp, pred)
        print(f'  → {pred}  |  {len(tokens)} tokens')
        plot_single_sequence(name, seq, 'ESM-2', tokens, imp, pred, OUTPUT_DIR)

        hydro = get_physicochemical_profile(tokens, HYDROPHOBICITY)
        charge = get_physicochemical_profile(tokens, CHARGE)
        r_h, p_h = pearsonr(imp, hydro) if len(np.unique(hydro)) > 1 else (0, 1)
        r_c, p_c = pearsonr(imp, charge) if len(np.unique(charge)) > 1 else (0, 1)
        summary_rows.append({
            'Model': 'ESM-2', 'Sequence': name, 'Prediction': pred,
            'r_hydro': round(r_h, 3), 'p_hydro': round(p_h, 4),
            'r_charge': round(r_c, 3), 'p_charge': round(p_c, 4),
        })

    del esm2_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtBERT ────────────────────────────────────────────────────────')
    pb_tok = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    pb_mdl = load_model(ProtBERTClassifier(), PROTBERT_PTH, 'ProtBERT')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        tokens, imp, pred = get_attention_and_tokens(seq, pb_mdl, pb_tok, 'bert')
        all_results[name]['ProtBERT'] = (tokens, imp, pred)
        print(f'  → {pred}  |  {len(tokens)} tokens')
        plot_single_sequence(name, seq, 'ProtBERT', tokens, imp, pred, OUTPUT_DIR)

        hydro = get_physicochemical_profile(tokens, HYDROPHOBICITY)
        charge = get_physicochemical_profile(tokens, CHARGE)
        r_h, p_h = pearsonr(imp, hydro) if len(np.unique(hydro)) > 1 else (0, 1)
        r_c, p_c = pearsonr(imp, charge) if len(np.unique(charge)) > 1 else (0, 1)
        summary_rows.append({
            'Model': 'ProtBERT', 'Sequence': name, 'Prediction': pred,
            'r_hydro': round(r_h, 3), 'p_hydro': round(p_h, 4),
            'r_charge': round(r_c, 3), 'p_charge': round(p_c, 4),
        })

    del pb_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtT5 ──────────────────────────────────────────────────────────')
    t5_tok = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    t5_mdl = load_model(ProtT5Classifier(freeze_t5=True), PROTT5_PTH, 'ProtT5')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        tokens, imp, pred = get_attention_and_tokens(seq, t5_mdl, t5_tok, 't5')
        all_results[name]['ProtT5'] = (tokens, imp, pred)
        print(f'  → {pred}  |  {len(tokens)} tokens')
        plot_single_sequence(name, seq, 'ProtT5', tokens, imp, pred, OUTPUT_DIR)

        hydro = get_physicochemical_profile(tokens, HYDROPHOBICITY)
        charge = get_physicochemical_profile(tokens, CHARGE)
        r_h, p_h = pearsonr(imp, hydro) if len(np.unique(hydro)) > 1 else (0, 1)
        r_c, p_c = pearsonr(imp, charge) if len(np.unique(charge)) > 1 else (0, 1)
        summary_rows.append({
            'Model': 'ProtT5', 'Sequence': name, 'Prediction': pred,
            'r_hydro': round(r_h, 3), 'p_hydro': round(p_h, 4),
            'r_charge': round(r_c, 3), 'p_charge': round(p_c, 4),
        })

    del t5_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── COMPARISON - all three models ──────────────────────────────────────')
    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        plot_three_model_comparison(name, seq, all_results[name], OUTPUT_DIR)

    plot_summary_heatmap(summary_rows, OUTPUT_DIR)

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'physicochemical_summary.csv')
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved at: {csv_path}')

    print('DONE!')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
