# 🌿 EWC Crop Disease Classification

A **continual learning** framework for crop disease classification using **Elastic Weight Consolidation (EWC)** to prevent catastrophic forgetting across seasonal tasks.

---

## 📌 Overview

This project applies **Elastic Weight Consolidation (EWC)** — a technique from neuroscience-inspired deep learning — to train a crop disease classifier **sequentially across seasons** (Spring → Summer → Autumn) without forgetting what it learned before.

The model uses a **pretrained ResNet-50 backbone** and a custom classification head, and is evaluated using **Backward Transfer (BWT)** to measure how well it retains knowledge from previous tasks.

---

## 🧠 Key Concepts

| Concept | Description |
|---|---|
| **EWC** | Elastic Weight Consolidation — penalizes large changes to weights that were important for previous tasks |
| **Fisher Information Matrix** | Used to estimate the importance of each parameter for a past task |
| **Backward Transfer (BWT)** | Measures how learning new tasks affects accuracy on previously learned tasks |
| **Continual Learning** | Training a model on a sequence of tasks without forgetting earlier ones |
| **Seasonal Tasks** | Spring, Summer, and Autumn each have different image augmentation profiles simulating real-world distribution shifts |

---

## 🗂️ Project Structure

```
ewc_crop_disease/
│
├── model.py          # ResNet-50 backbone + classification head
├── ewc_utils.py      # EWC class: Fisher matrix computation, penalty & loss
├── data_loader.py    # Seasonal dataset loaders (real or synthetic)
├── train.py          # Main training script with sequential EWC training
├── requirements.txt  # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

### 1. Model Architecture (`model.py`)
- **Backbone**: Pretrained `ResNet-50` (ImageNet weights), with optional early layer freezing
- **Head**: `Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.4) → Linear(512 → num_classes)`

### 2. Data Loading (`data_loader.py`)
- Supports **real datasets** (ImageFolder format under `./data/train` and `./data/val`)
- Falls back to **`FakeCropDataset`** (synthetic data) automatically if no real data is found
- Each season has a unique augmentation profile:
  - 🌸 **Spring**: Gamma boost + Gaussian noise
  - ☀️ **Summer**: Saturation boost + green channel enhancement
  - 🍂 **Autumn**: Desaturation + warm red-channel shift

### 3. EWC (`ewc_utils.py`)
- After each task, the **Fisher Information Matrix** is computed via sampled log-likelihood gradients
- The optimal weights `θ*` are saved
- During the next task, an **EWC penalty** is added to the loss:

```
Loss = CrossEntropy + (λ/2) × Σ F_i × (θ_i - θ*_i)²
```

### 4. Training (`train.py`)
- Trains tasks **sequentially**: Spring → Summer → Autumn
- EWC is applied from the **second task onwards**
- Uses **Adam optimizer** with **Cosine Annealing LR scheduler**
- Evaluates **Backward Transfer (BWT)** at the end

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ewc-crop-disease.git
cd ewc-crop-disease

# Install dependencies
pip install -r requirements.txt
```

### Running Training

```bash
python train.py
```

### Training with Custom Arguments

```bash
python train.py \
  --data_root ./data \
  --save_dir ./checkpoints \
  --epochs 15 \
  --batch_size 32 \
  --lr 5e-4 \
  --lambda_ewc 500.0 \
  --num_classes 10 \
  --fisher_samples 512 \
  --num_workers 4 \
  --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `./data` | Path to dataset |
| `--save_dir` | `./checkpoints` | Directory to save model checkpoints |
| `--epochs` | `15` | Number of epochs per task |
| `--batch_size` | `32` | Batch size |
| `--lr` | `5e-4` | Learning rate |
| `--lambda_ewc` | `500.0` | EWC regularization strength |
| `--num_classes` | `10` | Number of disease classes |
| `--fisher_samples` | `512` | Samples used to compute Fisher matrix |
| `--num_workers` | `4` | DataLoader worker threads |
| `--no_pretrain` | `False` | Disable ImageNet pretrained weights |
| `--seed` | `42` | Random seed for reproducibility |

---

## 📂 Dataset Format (for Real Data)

Organize your dataset as:

```
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── val/
    ├── class_0/
    ├── class_1/
    └── ...
```

> ⚠️ If no real dataset is found, the project automatically uses **synthetic data** for pipeline validation.

---

## 📊 Output & Results

After training, the following are saved to `./checkpoints/`:

- `model_Spring.pt` — checkpoint after Spring task
- `model_Summer.pt` — checkpoint after Summer task
- `model_final.pt` — final model after all tasks
- `results.json` — full training log including BWT score

### Example Results Summary (`results.json`)
```json
{
  "lambda_ewc": 500.0,
  "epochs": 15,
  "R_tt": { "Spring": 92.5, "Summer": 89.3, "Autumn": 87.1 },
  "R_Tt": { "Spring": 90.1, "Summer": 88.7, "Autumn": 87.1 },
  "BWT": -1.95
}
```

> ✅ **BWT ≥ -2.9%** → Catastrophic forgetting criterion **PASSED**

---

## 🧪 Evaluation Metric: Backward Transfer (BWT)

```
BWT = (1 / T-1) × Σ (R_Tt - R_tt)
```

- `R_tt` = accuracy on task *t* right after training on it
- `R_Tt` = accuracy on task *t* after training all tasks
- **Higher BWT (closer to 0) = less forgetting**

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Kirkpatrick et al. (2017) — *Overcoming catastrophic forgetting in neural networks*](https://arxiv.org/abs/1612.00796) — the original EWC paper
- [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/)
