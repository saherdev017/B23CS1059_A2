
# CSL 7640: Natural Language Understanding — Assignment 2

**Name:** Saher Dev
**Roll No:** B23CS1059
**Submitted to:** Prof. Anand Mishra

---

## Repository Structure

```
.
├── README.md
├── combined_report.pdf          # Final report (Problem 1 + Problem 2)
├── Problem1_Word2Vec.ipynb      # Notebook for Problem 1
├── Problem2_NameGeneration.ipynb # Notebook for Problem 2
├── corpus.txt           # Cleaned corpus (Problem 1 output)
├── TrainingNames.txt            # 1000 generated Indian names (Problem 2 input)
```

---

## Problem 1: Learning Word Embeddings from IIT Jodhpur Data

### Objective

Train Word2Vec models (CBOW and Skip-gram with Negative Sampling) from scratch on text scraped from IIT Jodhpur web pages and analyze the learned semantic structure.

### Data Sources

1. CSE Department Page — `https://www.iitj.ac.in/computer-science-engineering`
2. CSE Faculty Page — `https://www.iitj.ac.in/computer-science-engineering/en/faculty`
3. List of Academic Programs — `https://iitj.ac.in/office-of-academics/en/list-of-academic-programs`
4. Academic Regulations (mandatory) — `https://iitj.ac.in/office-of-academics/en/academic-regulations`

### Preprocessing

- HTML fetched via `requests`, parsed with `BeautifulSoup`
- Boilerplate removal (scripts, styles, headers, footers)
- Lowercasing
- Regex cleaning (`[^a-z\s]` removes non-alphabetical characters)
- Tokenization via `nltk.word_tokenize`
- Stop-word removal via `nltk.corpus.stopwords`
- Tokens of length ≤ 1 discarded

### Dataset Statistics

| Metric | Value |
|---|---|
| Documents | 4 |
| Total Tokens | 10,725 |
| Vocabulary Size | 1,871 |

### Models

Both models implemented **from scratch** in PyTorch using dual embedding matrices and a custom `NegativeSamplingLoss` (LogSigmoid + `torch.bmm`). No `nn.Linear` + `CrossEntropyLoss` shortcut — true negative sampling as per Mikolov et al.

**Baseline hyperparameters:** Embedding dim = 50, Context window = 2, Negative samples = 5, Batch size = 128, LR = 0.001, Optimizer = Adam

### Hyperparameter Experiments

| Configuration | Skip-gram Loss | CBOW Loss |
|---|---|---|
| Dim=50, Win=2, Neg=5 (Baseline) | 1.9429 | 6.1962 |
| Dim=100, Win=2, Neg=5 | **1.8615** | 7.6648 |
| Dim=50, Win=4, Neg=10 | 2.4052 | **4.0882** |

### Semantic Analysis

- **Nearest neighbors** reported for: `research`, `student`, `phd`, `exam`
- **Three analogy experiments:** UG:BTech::PG:?, Student:Study::Faculty:?, Exam:Grade::Research:?
- High cosine similarities (0.97–1.0) explained by small, domain-specific corpus

### Visualizations

- t-SNE projections for both Skip-gram and CBOW
- PCA projection for Skip-gram
- Cosine similarity heatmap

---

## Problem 2: Character-Level Name Generation Using RNN Variants

### Objective

Design and compare three recurrent architectures for character-level Indian name generation.

### Dataset

1,000 unique Indian names generated programmatically via Cartesian product of 33 prefixes × 28 roots × 19 suffixes, filtered to 4–10 characters, saved to `TrainingNames.txt`. Character vocabulary size: 22.

### Models

All three models implemented from scratch in PyTorch.

| Model | Parameters | Key Feature |
|---|---|---|
| Vanilla RNN | 8,406 | Single-layer Elman RNN |
| Bidirectional LSTM | 53,718 | Prefix-based training to prevent data leakage |
| RNN + Bahdanau Attention | 65,047 | Additive attention over past hidden states |

**Hyperparameters:**

- Vanilla RNN / BLSTM: Embedding=32, Hidden=64, LR=0.005, Epochs=20
- Attention RNN: Embedding=64, Hidden=128, LR=0.002, Epochs=20, Dropout=0.2

### Quantitative Results (500 generated names each)

| Metric | Vanilla RNN | BLSTM | Attention RNN |
|---|---|---|---|
| Diversity (%) | 92.00 | 80.40 | 80.60 |
| Novelty Rate (%) | 92.40 | 47.60 | 85.00 |
| Temperature | 0.8 | 0.8 | 0.7 |

### Key Findings

- **Vanilla RNN:** High novelty but phonetically incoherent outputs (no gating → vanishing gradients)
- **BLSTM:** Realistic names but heavy memorization (47.6% novelty); prefix-based training prevents leakage
- **Attention RNN:** Best balance of novelty (85%) and phonetic realism; Bahdanau attention lets the model refer back to earlier characters for coherent suffix generation

---

## How to Run

### Requirements

```
torch
numpy
matplotlib
nltk
beautifulsoup4
requests
scikit-learn
seaborn
wordcloud
```

### Problem 1

```bash
jupyter notebook Problem1_Word2Vec.ipynb
# Or run cells sequentially — fetches data from IIT Jodhpur URLs,
# trains both models, runs semantic analysis and generates all plots.
```

### Problem 2

```bash
jupyter notebook Problem2_NameGeneration.ipynb
# Generates dataset, trains all three models (Vanilla RNN, BLSTM, Attention),
# evaluates diversity/novelty, and produces loss comparison plot.
```

> **Note:** Final loss values and analogy outputs may vary slightly between runs due to random weight initialization, epoch shuffling, and dynamic negative sampling. This is expected behavior.

---

## Deliverables Checklist

### Problem 1
- [x] Source code (well-documented notebook)
- [x] Cleaned corpus file (`corpus.txt`)
- [x] Visualizations (word cloud, t-SNE, PCA, heatmap)
- [x] Report (in `report.pdf`)

### Problem 2
- [x] Source code for all three models
- [x] Generated name samples (printed in notebook + report)
- [x] Evaluation scripts (diversity & novelty computation)
- [x] Report (in `eport.pdf`)
