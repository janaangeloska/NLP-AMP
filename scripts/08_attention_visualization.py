import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
OUTPUT_DIR = '../results/attention_viz'

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


class ProtBERTClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            'Rostlab/prot_bert',
            output_attentions=True
        )
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
        # Mean pooling - ProtT5 does not have [CLS] tokens
        mask_exp = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        logits = self.classifier(self.dropout(mean_emb))
        return logits, outputs.attentions


def get_attentions(sequence, model, tokenizer, model_type):
    if model_type == 'bert':
        formatted = ' '.join(list(sequence))
    elif model_type == 't5':
        seq = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        formatted = f"<AA2fold> {' '.join(list(seq))}"
    else:
        formatted = sequence

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

    # Stack: (layers, heads, L, L)
    attn_stack = torch.stack(attentions, dim=0).squeeze(1).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    pred_label = 'AMP' if torch.argmax(logits, dim=1).item() == 1 else 'non-AMP'

    return attn_stack, tokens, pred_label


def aggregate_attention(attn_stack):
    num_layers = attn_stack.shape[0]
    split = max(1, num_layers // 3)
    per_layer = attn_stack.mean(axis=1)
    return (
        per_layer.mean(axis=0),
        per_layer[:split].mean(axis=0),
        per_layer[split * 2:].mean(axis=0),
    )


def compute_entropy(attn_stack):
    per_layer = attn_stack.mean(axis=1)
    eps = 1e-9
    return float(-np.sum(per_layer * np.log(per_layer + eps), axis=-1).mean())


def position_importance(mean_all, tokens):
    keep = [i for i, t in enumerate(tokens)
            if t not in SPECIAL_TOKENS and len(t.strip()) > 0]
    if not keep:
        return [], np.array([])
    clean = [tokens[i] for i in keep]
    imp = mean_all[np.ix_(keep, keep)].sum(axis=0)
    return clean, imp


def _bar_colors(importance):
    return [
        '#e74c3c' if v > np.percentile(importance, 75)
        else '#3498db' if v < np.percentile(importance, 25)
        else '#95a5a6'
        for v in importance
    ]


def plot_attention_heatmap(attn_matrix, tokens, title, ax, cmap='Blues'):
    keep = [i for i, t in enumerate(tokens)
            if t not in SPECIAL_TOKENS and len(t.strip()) > 0]
    if not keep:
        return
    mat = attn_matrix[np.ix_(keep, keep)]
    tok = [tokens[i] for i in keep]
    if len(tok) > 30:
        mat, tok = mat[:30, :30], tok[:30]

    sns.heatmap(mat, xticklabels=tok, yticklabels=tok,
                ax=ax, cmap=cmap, vmin=0, cbar=True,
                linewidths=0.1, linecolor='gray')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Клуч (Key)', fontsize=9)
    ax.set_ylabel('Прашувач (Query)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)


def safe_name(s):
    return s[:15].replace(' ', '_').replace('(', '').replace(')', '')


def plot_sequence_analysis(sequence, seq_name, model_name,
                           attn_stack, tokens, pred_label, output_dir):
    mean_all, mean_early, mean_late = aggregate_attention(attn_stack)
    num_layers = attn_stack.shape[0]
    split = max(1, num_layers // 3)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'{model_name}  |  {seq_name}  |  Prediction: {pred_label}\n{sequence}',
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    plot_attention_heatmap(mean_early, tokens,
                           f'Early layers (1–{split})',
                           fig.add_subplot(gs[0, 0]), 'Blues')
    plot_attention_heatmap(mean_late, tokens,
                           f'Late layers ({split * 2 + 1}–{num_layers})',
                           fig.add_subplot(gs[0, 1]), 'Reds')
    plot_attention_heatmap(mean_all, tokens,
                           'Average - all layers',
                           fig.add_subplot(gs[0, 2]), 'Purples')

    ax4 = fig.add_subplot(gs[1, 0])
    eps = 1e-9
    ents = [-np.sum(l * np.log(l + eps), axis=-1).mean()
            for l in attn_stack.mean(axis=1)]
    ax4.plot(range(1, len(ents) + 1), ents, 'o-',
             color='steelblue', linewidth=2, markersize=6)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Average entropy')
    ax4.set_title('Entropy per layer\n(lower = more focused)', fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_xticks(range(1, len(ents) + 1))

    ax5 = fig.add_subplot(gs[1, 1:])
    clean_tok, imp = position_importance(mean_all, tokens)
    if len(clean_tok) > 0:
        ax5.bar(range(len(clean_tok)), imp, color=_bar_colors(imp))
        ax5.set_xticks(range(len(clean_tok)))
        ax5.set_xticklabels(clean_tok, fontsize=8)
        ax5.set_xlabel('Amino acid')
        ax5.set_ylabel('Total attention (column sum)')
        ax5.set_title('Importance by position\n(red = top 25%, blue = bottom 25%)',
                      fontsize=10)
        ax5.grid(axis='y', alpha=0.3)

    path = os.path.join(output_dir,
                        f'attention_{model_name.lower().replace("-", "_")}_{safe_name(seq_name)}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_three_model_comparison(seq_name, sequence, results, output_dir):
    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.suptitle(
        f'ESM-2 vs ProtBERT vs ProtT5 - importance by position\n{seq_name}: {sequence}',
        fontsize=12, fontweight='bold'
    )

    for ax, model_name in zip(axes, models):
        data = results.get(model_name)
        if data is None:
            ax.set_visible(False)
            continue
        attn_stack, tokens, pred_label = data
        mean_all, _, _ = aggregate_attention(attn_stack)
        clean_tok, imp = position_importance(mean_all, tokens)
        if len(clean_tok) == 0:
            continue

        imp = imp / imp.sum()
        ax.bar(range(len(clean_tok)), imp, color=_bar_colors(imp))
        ax.set_xticks(range(len(clean_tok)))
        ax.set_xticklabels(clean_tok, fontsize=8)
        ax.set_title(f'{model_name}\nPrediction: {pred_label}', fontsize=11)
        ax.set_ylabel('Normalized importance')
        ax.set_xlabel('Amino acid')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'comparison_3models_{safe_name(seq_name)}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def load_model(model_obj, pth_path, label):
    if USE_SAVED_WEIGHTS and os.path.exists(pth_path):
        model_obj.load_state_dict(torch.load(pth_path, map_location=device))
        print(f'Weights: {pth_path}')
    else:
        print(f'WARNING: {pth_path} does not exist - pretrained weights')
    model_obj.eval().to(device)
    print(f'{label} is ready.')
    return model_obj


if __name__ == '__main__':
    print('ATTENTION VISUALIZATION - ProtBERT · ESM-2 · ProtT5')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {name: {} for name in TEST_SEQUENCES}

    print('\n── ESM-2 ──────────────────────────────────────────────────────────')
    esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm2_mdl = load_model(ESM2Classifier(), ESM2_PTH, 'ESM-2')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        attn, tok, pred = get_attentions(seq, esm2_mdl, esm2_tok, 'esm')
        all_results[name]['ESM-2'] = (attn, tok, pred)
        print(f'  → {pred}  |  shape {attn.shape}')
        plot_sequence_analysis(seq, name, 'ESM-2', attn, tok, pred, OUTPUT_DIR)

    del esm2_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtBERT ────────────────────────────────────────────────────────')
    pb_tok = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    pb_mdl = load_model(ProtBERTClassifier(), PROTBERT_PTH, 'ProtBERT')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        attn, tok, pred = get_attentions(seq, pb_mdl, pb_tok, 'bert')
        all_results[name]['ProtBERT'] = (attn, tok, pred)
        print(f'  → {pred}  |  shape {attn.shape}')
        plot_sequence_analysis(seq, name, 'ProtBERT', attn, tok, pred, OUTPUT_DIR)

    del pb_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── ProtT5 ──────────────────────────────────────────────────────────')
    t5_tok = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False
    )
    # freeze_t5=True -> Veltri
    t5_mdl = load_model(ProtT5Classifier(freeze_t5=True), PROTT5_PTH, 'ProtT5')

    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        attn, tok, pred = get_attentions(seq, t5_mdl, t5_tok, 't5')
        all_results[name]['ProtT5'] = (attn, tok, pred)
        print(f'  → {pred}  |  shape {attn.shape}')
        plot_sequence_analysis(seq, name, 'ProtT5', attn, tok, pred, OUTPUT_DIR)

    del t5_mdl
    torch.cuda.empty_cache()
    gc.collect()

    print('\n── COMPARISON - all three models ──────────────────────────────────────')
    for name, seq in TEST_SEQUENCES.items():
        print(f'\n  {name}')
        plot_three_model_comparison(name, seq, all_results[name], OUTPUT_DIR)

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    rows = []
    for name, seq in TEST_SEQUENCES.items():
        row = {'sequence_name': name, 'sequence': seq}
        for mdl, data in all_results[name].items():
            attn, _, pred = data
            row[f'{mdl}_prediction'] = pred
            row[f'{mdl}_entropy'] = round(compute_entropy(attn), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'attention_summary.csv')
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved at: {csv_path}')

    print('DONE!')
    print(f'End  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
