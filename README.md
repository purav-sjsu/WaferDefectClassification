# Wafer Defect Pattern Classification

In this project, we trained and evaluated various machine learning models to classify defect patterns on semiconductor wafer.

## Dataset

We used the [WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) dataset for this project

- 811,457 wafer samples from real fabs (172,950 labeled; 638,507 unlabeled)
- Each wafer is a 2-D matrix
(0: outside, 1: good, 2: defective)
- 9 classes:
`None`, `Edge-Ring`, `Edge-Loc`, `Center`, `Loc`, `Scratch`, `Random`, `Donut`, `Near-Full`

## Setup

### Step 1 - Install Dependencies

#### Option A: using uv (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and launch Jupyter
uv sync --dev
uv run jupyter notebook
```

#### Option B: using pip

```bash
python -m venv .venv
```

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Step 3 — Run Notebooks

```
uv run jupyter notebook
```


## Notebooks
1. `wafer_defects.ipynb` — Exploratory Data Analysis
2. `classical_ml.ipynb` — Classical ML experiments
3. `CNN.ipynb` — CNN experiments
4. `Autoencoders.ipynb` - Autoencoder experiment
4. `ResNet.ipynb` — ResNet experiment
5. `results.ipynb` — Compare trained models