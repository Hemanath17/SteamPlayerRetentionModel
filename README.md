# Steam Player Retention Model

A machine learning project that predicts whether Steam game players will continue playing after writing a review, using binary classification to forecast post-review engagement.

##  Project Objectives

- **Predict Player Engagement**: Determine if a player's playtime will increase after writing a Steam review
- **Identify Retention Patterns**: Understand which review characteristics correlate with continued gameplay
- **Support Business Decisions**: Enable game developers and platforms to identify highly engaged players and predict churn risk

##  Goals

1. Build a production-ready ML model with **83.6% accuracy** and **0.91 F1-score**
2. Deploy an end-to-end MLOps pipeline on DigitalOcean cloud infrastructure
3. Create a scalable model serving API for real-time predictions
4. Implement proper model versioning and experiment tracking with MLflow

##  Architecture

The project follows a cloud-native MLOps architecture deployed on DigitalOcean:

![Cloud Architecture](cloud_architecture.png)

### Key Components

- **Data Layer**: PostgreSQL database storing cleaned Steam review data
- **Application Layer**: DigitalOcean Droplet running:
  - MLflow Tracking Server (experiment tracking & model registry)
  - Model Training Pipeline (feature engineering, training, evaluation)
  - Inference Server (REST API for predictions)
- **Storage**: DigitalOcean Spaces (S3-compatible) for model artifacts
- **Infrastructure**: Terraform for automated infrastructure provisioning

##  Dataset

The model uses Steam review data from three popular games:
- Cyberpunk 2077
- Red Dead Redemption 2
- The Witcher 3

**Features Used:**
- **Numeric**: Playtime metrics, votes, comments, review statistics, text metrics
- **Boolean**: Purchase type, early access, Steam Deck usage
- **Categorical**: Language, sentiment labels

**Target Variable**: Binary classification
- `1` = Player's playtime increases after review
- `0` = No increase in playtime

## 🤖 Model

**Algorithm**: `HistGradientBoostingClassifier` (scikit-learn)

**Why This Model:**
- Handles non-linear relationships between features
- Robust to missing values and outliers
- Efficient for large datasets
- Provides feature importance insights

**Preprocessing Pipeline:**
- StandardScaler for numeric features
- OneHotEncoder for categorical features
- Passthrough for boolean features

**Performance Metrics:**
- **Accuracy**: 83.6%
- **F1-Score**: 0.91 (for positive class)
- **Precision**: 84.4% (increase class)
- **Recall**: 98.6% (increase class)

##  Process Workflow

### 1. Data Preparation
```bash
python -m steam_model.load_to_db
```
- Loads Excel files containing Steam reviews
- Cleans and validates data
- Creates binary target variable
- Stores in PostgreSQL database

### 2. Model Training
```bash
python -m steam_model.train_steam_model
```
- Loads data from PostgreSQL
- Performs group-based train/validation/test split (by player ID)
- Trains HistGradientBoostingClassifier
- Logs metrics and model to MLflow
- Registers model in MLflow Model Registry

### 3. Model Deployment
```bash
docker-compose up -d
```
- Starts MLflow Tracking Server (port 5050)
- Starts Inference Server (port 5001)
- Serves model via REST API

### 4. Making Predictions
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

##  Cloud Services

- **DigitalOcean PostgreSQL**: Data storage
- **DigitalOcean Spaces**: Model artifact storage (S3-compatible)
- **DigitalOcean Droplet**: Compute for MLflow and inference
- **Docker Hub**: Container registry

##  Setup & Installation

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- PostgreSQL database (DigitalOcean)
- DigitalOcean Spaces bucket
- Terraform (for infrastructure)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Hemanath17/SteamPlayerRetentionModel.git
cd SteamPlayerRetentionModel
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Create a `.env` file:
```env
APP_DB_URL=postgresql://user:password@host:port/database
MLFLOW_TRACKING_URI=http://localhost:5050
AWS_ACCESS_KEY_ID=your_spaces_key
AWS_SECRET_ACCESS_KEY=your_spaces_secret
AWS_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
```

4. **Deploy infrastructure** (optional)
```bash
cd droplet_project
terraform init
terraform apply
```

5. **Start services**
```bash
docker-compose up -d
```

## 📁 Project Structure

```
SteamPlayerRetentionModel/
├── steam_model/              # Core ML code
│   ├── load_to_db.py        # Data loading script
│   ├── train_steam_model.py # Model training script
│   └── __init__.py
├── droplet_project/         # Terraform infrastructure
│   ├── main.tf
│   └── provider.tf
├── docker-compose.yml        # Service orchestration
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
└── cloud_architecture.png  # Architecture diagram
```

## 🔬 Model Evaluation

The model uses **GroupShuffleSplit** to prevent data leakage:
- Splits data by `author_steamid` (player ID)
- Ensures same player doesn't appear in both train and test sets
- Train: 60%, Validation: 20%, Test: 20%

**Baseline Comparison:**
- Model outperforms majority-class baseline
- Demonstrates meaningful predictive power beyond naive predictions

## 📈 Business Value

- **Early Engagement Detection**: Identify highly engaged players
- **Churn Prediction**: Flag players likely to stop playing
- **Personalized Interventions**: Target at-risk players with retention strategies
- **Review Analysis**: Understand which review characteristics predict continued engagement


