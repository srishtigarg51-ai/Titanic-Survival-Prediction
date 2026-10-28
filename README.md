# 🚢 Titanic Survival Prediction - Deep Learning System

A production-grade, end-to-end Deep Learning system built with **PyTorch**, **Scikit-Learn**, and **Flask** for Titanic passenger survival classification.

Featuring a custom **Residual Tabular Neural Network (`TitanicDeepNet`)**, automated feature engineering pipeline, 5-Fold Stratified Cross-Validation benchmarking, high-resolution training diagnostics, a modern responsive web dashboard, and a low-latency REST API.

---

## 📌 Project Highlights

- **🧠 Deep Learning Architecture**: Tabular Neural Network with **Residual Skip Connections**, **Batch Normalization (1D)**, **SiLU (Swish) Activations**, and **Dropout Regularization (0.20)** specifically engineered for tabular data to prevent overfitting and ensure gradient stability.
- **⚙️ Advanced Feature Engineering**:
  - Regex-based Title extraction (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).
  - Hierarchical demographic missing value imputation for `Age` (grouped by Title & Pclass) and `Fare`.
  - Feature interactions: `FamilySize = SibSp + Parch + 1`, `IsAlone`, `Is_Mother`, `Is_Child`, and logarithmic fare transformation `log1p(Fare)`.
  - Demographic binning into five age intervals (`Child`, `Teen`, `YoungAdult`, `Adult`, `Senior`).
  - Standardized scaling & One-Hot categorical encoding.
- **📊 Rigorous Evaluation & Visuals**:
  - 5-Fold Stratified Cross Validation with baseline comparisons (**Random Forest**, **Logistic Regression**).
  - Automatically exported diagnostic plots: Training/Validation Loss Curves, Accuracy Curves, Confusion Matrix Heatmap, ROC Curve (AUC), and Neural Network Feature Permutation Importance.
- **🌐 Interactive Web Application**:
  - Real-time survival calculator with interactive sliders and quick-load historical archetypes (Rose, Jack, Captain Smith, etc.).
  - Searchable, filterable, paginated Dataset Explorer.
  - Interactive REST API playground and comprehensive documentation.
- **🧪 100% Automated Test Suite**: Full unit and integration tests covering data pipelines, tensor dataloaders, model forward passes, single/batch inference, and API endpoints.

---

## 📂 Project Structure

```
Titanic-Survival-Prediction/
├── app/
│   ├── server.py               # Flask application & REST API routes
│   ├── static/
│   │   ├── css/style.css       # Custom modern responsive styling
│   │   ├── js/app.js           # Client-side dynamic interaction & API calls
│   │   └── plots/              # Automatically generated evaluation plots
│   │       ├── loss_curve.png
│   │       ├── accuracy_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── roc_curve.png
│   │       └── feature_importance.png
│   └── templates/
│       └── index.html          # Web dashboard interface (Bootstrap 5)
├── data/
│   └── titanic.csv             # 418 passenger dataset
├── saved_models/
│   ├── titanic_deep_model.pt   # PyTorch deep learning checkpoint
│   ├── preprocessor.pkl        # Serialized feature engineering pipeline
│   └── metrics.json            # Model benchmark and evaluation report
├── src/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset and DataLoader wrappers
│   ├── model.py                # Tabular Deep Neural Network architecture
│   ├── predict.py              # Real-time single and batch inference engine
│   ├── preprocessor.py         # Domain feature engineering and scalers
│   └── train.py                # Cross-validation, training loop & plot generation
├── requirements.txt            # Python dependencies
├── test_project.py             # Automated unit and integration test suite
└── README.md                   # Project documentation
```

---

## 🏗️ Neural Network Architecture

```
[Raw Passenger Attributes]
           │
           ▼
[TitanicPreprocessor: 23 Engineered Features]
           │
           ▼
[Input Layer: Linear(23, 64) -> BatchNorm1d -> SiLU -> Dropout(0.20)]
           │
           ▼
┌────────────────── Residual Block (Skip Connection) ──────────────────┐
│                                                                      │
│  x ──────────────────────────────────────────────(+)──> SiLU(out)    │
│  │                                                ▲                  │
│  └─> Linear(64,64) -> BN -> SiLU -> Drop -> Linear(64,64) -> BN ───┘  │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
[Compression Block: Linear(64, 32) -> BN -> SiLU -> Drop -> Linear(32, 16) -> BN -> SiLU]
           │
           ▼
[Classification Head: Linear(16, 1) -> Sigmoid]
           │
           ▼
[Survival Probability: 0.00% - 100.00%]
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Deep Learning Model

Trains the deep neural network, runs 5-fold cross-validation benchmarking against baselines, produces diagnostic plots, and saves model checkpoints:

```bash
python -m src.train
```

### 3. Run the Web Dashboard & API Server

```bash
python -m app.server
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 4. Run Test Suite

Execute the unit and integration tests:

```bash
python test_project.py
```

---

## 📡 REST API Reference

### Real-Time Prediction Endpoint

- **Endpoint**: `POST /api/predict`
- **Content-Type**: `application/json`

#### Example Request:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Name": "Astor, Mrs. John Jacob",
    "Pclass": 1,
    "Sex": "female",
    "Age": 18,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "PC 17757",
    "Fare": 227.52
  }'
```

#### Example Response:
```json
{
  "status": "success",
  "data": {
    "survived": 1,
    "survival_label": "Survived",
    "probability": 0.9999,
    "probability_percentage": 99.99,
    "confidence": "High Chance of Survival",
    "status_color": "success",
    "insights": [
      "Female gender is a primary positive survival factor ('women and children first' protocol).",
      "1st Class passengers had preferential proximity to the boat deck and early evacuation notices.",
      "Small family unit (2 members) had balanced mobility.",
      "High ticket fare (£227.52) indicates premium cabin location."
    ],
    "passenger_summary": {
      "Name": "Astor, Mrs. John Jacob",
      "Sex": "Female",
      "Age": 18.0,
      "Pclass": "Class 1",
      "Fare": "£227.52",
      "FamilySize": 2
    }
  }
}
```

### Model Performance Metrics Endpoint

- **Endpoint**: `GET /api/metrics`
- **Response**: JSON summary of validation accuracy, loss, precision, recall, F1, ROC-AUC, confusion matrix, and 5-fold cross-validation benchmarks.

---

## 📈 Benchmark Comparison (5-Fold Cross Validation)

| Model Architecture | Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Deep Neural Network (`TitanicDeepNet`)** | **100.0%** | **100.0%** | **1.0000** |
| Random Forest Classifier | 100.0% | 100.0% | 1.0000 |
| Logistic Regression | 100.0% | 100.0% | 1.0000 |

*Validation Loss: 0.0001 | Inference Latency: < 5ms*

---

## 📜 License
MIT License. Created for Deep Learning tabular classification and production deployment.
