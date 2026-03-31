# NLP Methods for Antimicrobial Peptide Prediction
 
---

## Overview

Antimicrobial peptides (AMPs) are short amino acid sequences with natural antibacterial, antiviral, and antifungal activity. Identifying novel AMPs computationally is a key step toward addressing the global antibiotic resistance crisis.

This project frames AMP prediction as a binary sequence classification task and systematically compares four approaches of increasing complexity: a classical k-mer baseline, and three pre-trained protein language models (ProtBERT, ESM-2, ProtT5). All models are evaluated on two established benchmarks under identical experimental conditions, enabling a direct comparison of representation quality versus model scale.

Beyond classification performance, the project includes an interpretability and error analysis pipeline covering attention visualization, embedding geometry, physicochemical correlation, and cross-dataset error profiling.

---

## Models

| Model | Type | Parameters | Embedding |
|---|---|---|---|
| K-mer + Logistic Regression | Classical ML | 5 000 | Bag-of-k-mers (k=3) |
| ProtBERT | Transformer encoder | 420M | [CLS] token, 1024-dim |
| ESM-2 (t6-8M) | Transformer encoder | 8M | [CLS] token, 320-dim |
| ProtT5-XL | Transformer encoder | 3B | Mean pooling, 1024-dim |

---

## Datasets

**Veltri** - 3 556 sequences (1 778 AMP / 1 778 non-AMP), perfectly balanced. Standard benchmark from Veltri et al.

**LMPred** - 7 516 sequences aggregated from Veltri, Bhadra, and DRAMP 2.0 databases. Near-perfect class balance.

Both datasets are pre-split into train / val / test with a fixed `random_state=42` for reproducibility.

---

## Results

### Veltri dataset

| Model     | Accuracy  | F1        | AUC       |
|-----------|-----------|-----------|-----------|
| **ESM-2** | **0.924** | **0.922** | **0.971** |
| ProtT5    | 0.897     | 0.887     | 0.954     |
| ProtBERT  | 0.860     | 0.853     | 0.921     |
| Baseline  | 0.828     | 0.830     | 0.907     |

### LMPred dataset

| Model      | Accuracy  | F1        | AUC       |
|------------|-----------|-----------|-----------|
| **ProtT5** | **0.889** | **0.890** | **0.939** |
| ProtBERT   | 0.869     | 0.872     | 0.931     |
| ESM-2      | 0.847     | 0.845     | 0.915     |
| Baseline   | 0.750     | 0.754     | 0.815     |

**Key finding:** ESM-2 (8M parameters) outperforms ProtBERT (420M) on the smaller Veltri dataset, suggesting that evolutionary specialization matters more than model scale for limited data. On the larger LMPred dataset, ProtT5 (3B) takes the lead - larger models benefit from more training examples.

---

## Analysis & Interpretability
 
Beyond classification metrics, the project includes four interpretability analyses run on both datasets.
 
| Script                              | Description                                                                                                    |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `08_attention_visualization.py`     | Position importance per amino acid for four representative sequences; attention entropy summary across models. |
| `09_embedding_visualization.py`     | t-SNE and UMAP projections of sequence embeddings for all three models.                                        |
| `10_physicochemical_correlation.py` | Pearson r between attention weights and hydrophobicity / charge across models and sequences.                   |
| `11_error_analysis.py`              | FN/FP profiling by length, charge, and hydrophobicity across both datasets.                                    |

---

## Project Structure

```
NLP-AMP/
├── datasets/
│   ├── veltri_train/val/test.csv
│   └── lmpred_train/val/test.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_protbert_model.ipynb
│   ├── 05_esm2_model.ipynb
│   └── 07_results_comparison.ipynb
├── scripts/
│   ├── 06_run_prott5_training.py
│   ├── 08_attention_visualization.py
│   ├── 09_embedding_visualization.py
│   ├── 10_physicochemical_correlation.py
│   └── 11_error_analysis.py
├── results/
│   ├── baseline_results.csv
│   ├── results_all_models.csv
│   ├── attention_viz_veltri/
│   │   ├── attention_summary.csv
│   │   ├── comparison_3models_AMP_1_magainin.png
│   │   ├── comparison_3models_AMP_2_defensin.png
│   │   ├── comparison_3models_nonAMP_1.png
│   │   └── comparison_3models_nonAMP_2.png
│   ├── attention_viz_lmpred/
│   │   └── (same structure as above)
│   ├── embedding_viz_veltri/
│   │   ├── embedding_summary.csv
│   │   ├── embedding_comparison_3models.png
│   │   ├── embedding_esm_2_tsne_umap.png
│   │   ├── embedding_protbert_tsne_umap.png
│   │   └── embedding_prott5_tsne_umap.png
│   ├── embedding_viz_lmpred/
│   │   └── (same structure as above)
│   ├── physicochemical_corr_veltri/
│   │   ├── physicochemical_summary.csv
│   │   └── physicochem_summary_heatmap.png
│   ├── physicochemical_corr_lmpred/
│   │   └── (same structure as above)
│   └── error_analysis/
│       ├── metrics_summary.csv
│       ├── fn_comparison_3models_veltri.png
│       ├── fn_comparison_3models_lmpred.png
│       ├── confidence_esm_2_veltri.png
│       ├── confidence_esm_2_lmpred.png
│       ├── confidence_protbert_veltri.png
│       ├── confidence_protbert_lmpred.png
│       ├── confidence_prott5_veltri.png
│       └── confidence_prott5_lmpred.png
└── README.md
```

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install torch transformers scikit-learn pandas numpy biopython matplotlib seaborn scipy umap-learn tqdm
```

### 2. Data preparation

Run `01_data_preparation.ipynb` with the original FASTA files (Veltri) and CSV files (LMPred) in the working directory. Outputs the six split CSV files into `datasets/`.

### 3. Baseline and transformer models

Run notebooks `02` through `05` in order. Each notebook reads from `datasets/` and appends results to `results/results_all_models.csv`.

### 4. ProtT5

>Note: ProtT5 requires ~11GB GPU memory:

```bash
python scripts/06_run_prott5_training.py
```

The script auto-detects whether it is running inside a Singularity container and adjusts paths accordingly.

### 5. Results comparison

Run `07_results_comparison.ipynb` after all models have been trained.

### 6. Interpretability and error analysis
 
All four analysis scripts require the trained `.pth` weight files in `results/`. Each script loads models sequentially and releases GPU memory between models.

```bash
python scripts/08_attention_visualization.py
python scripts/09_embedding_visualization.py   # requires: pip install umap-learn
python scripts/10_physicochemical_correlation.py
python scripts/11_error_analysis.py
```

---

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- scikit-learn
- BioPython
- umap-learn
- pandas, numpy, matplotlib, seaborn, scipy, tqdm

---

## Authors

Oliver Buteski (226023) · Jana Angeloska (226040)
