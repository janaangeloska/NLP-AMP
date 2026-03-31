import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
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

TEST_CSV = '../data/veltri_test.csv'

USE_SAVED_WEIGHTS = True
OUTPUT_DIR = '../results/embedding_viz'

MAX_SAMPLES = None

TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProtBERTClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('Rostlab/prot_bert')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        return logits, cls


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
        logits = self.classifier(self.dropout(cls))
        return logits, cls


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
        # Mean pooling (ProtT5)
        mask_exp = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        mean_emb = torch.sum(embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        logits = self.classifier(self.dropout(mean_emb))
        return logits, mean_emb


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


def extract_embeddings(df, model, tokenizer, model_type, batch_size=32):
    all_embeddings = []
    all_preds = []

    sequences = df['sequence'].tolist()
    n = len(sequences)

    print(f'Extraction of {n} embeddings (batch={batch_size})...')

    for start in range(0, n, batch_size):
        batch_seqs = sequences[start: start + batch_size]
        formatted = [format_sequence(s, model_type) for s in batch_seqs]

        encoding = tokenizer(
            formatted,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits, emb = model(encoding['input_ids'], encoding['attention_mask'])

        all_embeddings.append(emb.cpu().numpy())
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())

        if (start // batch_size + 1) % 10 == 0:
            done = min(start + batch_size, n)
            print(f'{done}/{n} sequences processed')

    embeddings = np.vstack(all_embeddings)
    print(f'Embedding shape: {embeddings.shape}')
    return embeddings, np.array(all_preds)


def reduce_embeddings(embeddings):
    print('PCA (50 components)...')
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    n_components = min(50, emb_scaled.shape[0], emb_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(emb_scaled)

    print(f't-SNE (perplexity={TSNE_PERPLEXITY})...')
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_ITERATIONS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    emb_tsne = tsne.fit_transform(emb_pca)

    print(f'UMAP (n_neighbors={UMAP_NEIGHBORS})...')
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=RANDOM_STATE
    )
    emb_umap = reducer.fit_transform(emb_pca)

    return emb_tsne, emb_umap, emb_pca, pca


def compute_separation_metrics(emb_2d, true_labels, pred_labels):
    metrics = {}
    if len(np.unique(true_labels)) > 1:
        metrics['silhouette_true'] = round(
            silhouette_score(emb_2d, true_labels), 4)
    if len(np.unique(pred_labels)) > 1:
        metrics['silhouette_pred'] = round(
            silhouette_score(emb_2d, pred_labels), 4)
    correct = (true_labels == pred_labels).mean() * 100
    metrics['accuracy_pct'] = round(correct, 2)
    return metrics


AMP_COLOR = '#e74c3c'
NOAMP_COLOR = '#3498db'
WRONG_MARKER = 'X'


def _scatter(ax, coords, true_labels, pred_labels, title, show_errors=True):
    colors = [AMP_COLOR if l == 1 else NOAMP_COLOR for l in true_labels]
    correct = (true_labels == pred_labels)

    mask_c = correct
    ax.scatter(coords[mask_c, 0], coords[mask_c, 1],
               c=[colors[i] for i in np.where(mask_c)[0]],
               alpha=0.55, s=18, linewidths=0)

    if show_errors:
        mask_w = ~correct
        if mask_w.sum() > 0:
            ax.scatter(coords[mask_w, 0], coords[mask_w, 1],
                       c=[colors[i] for i in np.where(mask_w)[0]],
                       alpha=0.9, s=40, marker=WRONG_MARKER,
                       linewidths=0.8, edgecolors='black',
                       label='Misclassified')
            ax.legend(fontsize=8, loc='upper right')

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=AMP_COLOR, label='AMP (1)'),
        Patch(facecolor=NOAMP_COLOR, label='non-AMP (0)'),
    ]
    ax.legend(handles=handles, fontsize=9, loc='lower right')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=9)
    ax.set_ylabel('Dim 2', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)


def plot_single_model(model_name, emb_tsne, emb_umap,
                      true_labels, pred_labels,
                      metrics_tsne, metrics_umap, output_dir):
    """4-panel figure: t-SNE + UMAP × точно/погрешно."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'{model_name} - Embedding projection on dataset\n'
        f't-SNE silhouette: {metrics_tsne.get("silhouette_true", "N/A")} | '
        f'UMAP silhouette: {metrics_umap.get("silhouette_true", "N/A")} | '
        f'Accuracy: {metrics_tsne.get("accuracy_pct", "N/A")}%',
        fontsize=12, fontweight='bold', y=1.01
    )

    _scatter(axes[0], emb_tsne, true_labels, pred_labels,
             f't-SNE  (perplexity={TSNE_PERPLEXITY})')
    _scatter(axes[1], emb_umap, true_labels, pred_labels,
             f'UMAP  (n_neighbors={UMAP_NEIGHBORS})')

    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'embedding_{model_name.lower().replace("-", "_")}_tsne_umap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_three_model_comparison(model_results, true_labels, output_dir):
    models = ['ESM-2', 'ProtBERT', 'ProtT5']
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        'ESM-2 vs ProtBERT vs ProtT5 - Embedding spaces',
        fontsize=14, fontweight='bold', y=1.01
    )

    for col, mname in enumerate(models):
        if mname not in model_results:
            continue
        res = model_results[mname]
        sil = res['metrics_tsne'].get('silhouette_true', 'N/A')
        _scatter(axes[0, col], res['emb_tsne'], true_labels, res['pred_labels'],
                 f'{mname}\nt-SNE  sil={sil}', show_errors=False)
        sil_u = res['metrics_umap'].get('silhouette_true', 'N/A')
        _scatter(axes[1, col], res['emb_umap'], true_labels, res['pred_labels'],
                 f'{mname}\nUMAP  sil={sil_u}', show_errors=False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'embedding_comparison_3models.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_pca_variance(model_name, pca, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-',
            color='steelblue', linewidth=2, markersize=5)
    ax.axhline(90, color='red', linestyle='--', alpha=0.6, label='90% variance')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.6, label='95% variance')
    ax.set_xlabel('Number of PCA components', fontsize=10)
    ax.set_ylabel('Cumulative variance (%)', fontsize=10)
    ax.set_title(f'{model_name} - PCA Scree Plot', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'pca_variance_{model_name.lower().replace("-", "_")}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


def plot_embedding_distance_distribution(model_results, true_labels, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Intra-class vs Inter-class distances (UMAP space)',
                 fontsize=13, fontweight='bold')

    amp_idx = np.where(true_labels == 1)[0]
    noamp_idx = np.where(true_labels == 0)[0]

    for ax, (mname, res) in zip(axes, model_results.items()):
        coords = res['emb_umap']

        amp_s = amp_idx[:200]
        noamp_s = noamp_idx[:200]

        def pairwise_dist(idx_a, idx_b):
            a, b = coords[idx_a], coords[idx_b]
            dists = []
            for i in range(min(len(a), 200)):
                for j in range(min(len(b), 200)):
                    if idx_a is idx_b and j <= i:
                        continue
                    dists.append(np.linalg.norm(a[i] - b[j]))
            return np.array(dists)

        intra_amp = pairwise_dist(amp_s, amp_s)
        intra_noamp = pairwise_dist(noamp_s, noamp_s)
        inter = pairwise_dist(amp_s, noamp_s)

        ax.hist(intra_amp, bins=40, alpha=0.6, color=AMP_COLOR, label='AMP–AMP')
        ax.hist(intra_noamp, bins=40, alpha=0.6, color=NOAMP_COLOR, label='non-AMP–non-AMP')
        ax.hist(inter, bins=40, alpha=0.4, color='gray', label='AMP–non-AMP')
        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Euclidean distance', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'embedding_distance_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved at: {path}')


if __name__ == '__main__':
    print('EMBEDDING VISUALIZATION - ProtBERT · ESM-2 · ProtT5')
    print(f'Start : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'\nLoading: {TEST_CSV}')
    df = pd.read_csv(TEST_CSV)
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    true_labels = df['label'].values
    print(f'Total: {len(df)} sequences'
          f'(AMP={true_labels.sum()}, non-AMP={(true_labels == 0).sum()})')

    model_results = {}  # {model_name: {emb_tsne, emb_umap, pred_labels, metrics_*}}
    summary_rows = []

    print('\n── ESM-2 ──────────────────────────────────────────────────────────')
    esm2_tok = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    esm2_mdl = load_model(ESM2Classifier(), ESM2_PTH, 'ESM-2')

    embs, preds = extract_embeddings(df, esm2_mdl, esm2_tok, 'esm')
    del esm2_mdl
    torch.cuda.empty_cache()
    gc.collect()

    emb_tsne, emb_umap, _, pca_esm = reduce_embeddings(embs)
    mt = compute_separation_metrics(emb_tsne, true_labels, preds)
    mu = compute_separation_metrics(emb_umap, true_labels, preds)
    print(f't-SNE silhouette (true): {mt.get("silhouette_true")}')
    print(f'UMAP  silhouette (true): {mu.get("silhouette_true")}')
    print(f'Accuracy: {mt.get("accuracy_pct")}%')

    plot_single_model('ESM-2', emb_tsne, emb_umap, true_labels, preds, mt, mu, OUTPUT_DIR)
    plot_pca_variance('ESM-2', pca_esm, OUTPUT_DIR)

    model_results['ESM-2'] = dict(emb_tsne=emb_tsne, emb_umap=emb_umap,
                                  pred_labels=preds,
                                  metrics_tsne=mt, metrics_umap=mu)
    summary_rows.append({'Model': 'ESM-2', **mt,
                         'umap_silhouette_true': mu.get('silhouette_true'),
                         'umap_silhouette_pred': mu.get('silhouette_pred')})
    del embs

    print('\n── ProtBERT ────────────────────────────────────────────────────────')
    pb_tok = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    pb_mdl = load_model(ProtBERTClassifier(), PROTBERT_PTH, 'ProtBERT')

    embs, preds = extract_embeddings(df, pb_mdl, pb_tok, 'bert')
    del pb_mdl
    torch.cuda.empty_cache()
    gc.collect()

    emb_tsne, emb_umap, _, pca_pb = reduce_embeddings(embs)
    mt = compute_separation_metrics(emb_tsne, true_labels, preds)
    mu = compute_separation_metrics(emb_umap, true_labels, preds)
    print(f't-SNE silhouette (true): {mt.get("silhouette_true")}')
    print(f'UMAP  silhouette (true): {mu.get("silhouette_true")}')
    print(f'Accuracy: {mt.get("accuracy_pct")}%')

    plot_single_model('ProtBERT', emb_tsne, emb_umap, true_labels, preds, mt, mu, OUTPUT_DIR)
    plot_pca_variance('ProtBERT', pca_pb, OUTPUT_DIR)

    model_results['ProtBERT'] = dict(emb_tsne=emb_tsne, emb_umap=emb_umap,
                                     pred_labels=preds,
                                     metrics_tsne=mt, metrics_umap=mu)
    summary_rows.append({'Model': 'ProtBERT', **mt,
                         'umap_silhouette_true': mu.get('silhouette_true'),
                         'umap_silhouette_pred': mu.get('silhouette_pred')})
    del embs

    print('\n── ProtT5 ──────────────────────────────────────────────────────────')
    t5_tok = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False
    )
    t5_mdl = load_model(ProtT5Classifier(freeze_t5=True), PROTT5_PTH, 'ProtT5')

    embs, preds = extract_embeddings(df, t5_mdl, t5_tok, 't5')
    del t5_mdl
    torch.cuda.empty_cache()
    gc.collect()

    emb_tsne, emb_umap, _, pca_t5 = reduce_embeddings(embs)
    mt = compute_separation_metrics(emb_tsne, true_labels, preds)
    mu = compute_separation_metrics(emb_umap, true_labels, preds)
    print(f't-SNE silhouette (true): {mt.get("silhouette_true")}')
    print(f'UMAP silhouette (true): {mu.get("silhouette_true")}')
    print(f'Accuracy: {mt.get("accuracy_pct")}%')

    plot_single_model('ProtT5', emb_tsne, emb_umap, true_labels, preds, mt, mu, OUTPUT_DIR)
    plot_pca_variance('ProtT5', pca_t5, OUTPUT_DIR)

    model_results['ProtT5'] = dict(emb_tsne=emb_tsne, emb_umap=emb_umap,
                                   pred_labels=preds,
                                   metrics_tsne=mt, metrics_umap=mu)
    summary_rows.append({'Model': 'ProtT5', **mt,
                         'umap_silhouette_true': mu.get('silhouette_true'),
                         'umap_silhouette_pred': mu.get('silhouette_pred')})
    del embs

    print('\n── COMPARISON - all three models ──────────────────────────────────────')
    plot_three_model_comparison(model_results, true_labels, OUTPUT_DIR)
    plot_embedding_distance_distribution(model_results, true_labels, OUTPUT_DIR)

    print('\n── SUMMARY TABLE ──────────────────────────────────────────────────')
    summary_df = pd.DataFrame(summary_rows).rename(columns={
        'silhouette_true': 'tsne_silhouette_true',
        'silhouette_pred': 'tsne_silhouette_pred',
        'accuracy_pct': 'accuracy_pct'
    })
    csv_path = os.path.join(OUTPUT_DIR, 'embedding_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(summary_df.to_string(index=False))
    print(f'\nSaved at: {csv_path}')

    print('DONE!')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Output: {OUTPUT_DIR}')
