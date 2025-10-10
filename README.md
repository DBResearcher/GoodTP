# GoodTP

An effective data selection framework for trajectory similarity learning framework supporting multiple models (T3S, TrjSR, TrajGAT, TMN) and similarity measures (DTW, Hausdorff, etc.).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)

## Prerequisites

- Python 3.9+
- Redis server
- CUDA-capable GPU (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd GoodTP
```

### 2. Set Up Python Environment

```bash
# Create a conda environment (recommended)
conda create -n goodtp python=3.9
conda activate goodtp

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Additional Dependencies

#### traj_dist Package

Install the trajectory distance computation package manually:

```bash
# Follow instructions at: https://github.com/bguillouet/traj-dist
pip install git+https://github.com/bguillouet/traj-dist.git
```

#### PyTorch

Install PyTorch according to your CUDA version:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Set Up Redis Database

Redis is used for caching trajectory similarity computations.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Verify Redis is running:**
```bash
redis-cli ping
# Should return: PONG
```

### 5. Start Trajectory Similarity API Service

The FastAPI service handles trajectory similarity computations on port 8800.

```bash
cd trajsimi_service
bash run.sh
```

**Verify the service is running:**
```bash
curl http://localhost:8800/docs
```

You should see the FastAPI documentation page.

## Data Preparation

Demo datasets are provided in the `data/` directory due to the GitHub's file size limit of 100.00 MB:

- **Porto dataset:** `data/porto_inf_trajsimi_dict_traj_demo`
- **Chengdu dataset:** `data/chengdu_inf_trajsimi_dict_traj_demo`

For full datasets, please refer to the original data sources or contact the authors.

## Quick Start

All training scripts are located in the `script/` directory.

### Basic Training Command

```bash
cd script
python <MODEL>_train_script.py --cluster_num <NUM> --num_iterations <ITER> --budget <BUDGET> --dataset <DATASET> --trajsimi_measure <MEASURE>
```

## Usage Examples

### Example 1: T3S Model on Porto with DTW (without weight)

```bash
cd script
python T3S_train_script.py \
    --cluster_num 10 \
    --num_iterations 100 \
    --budget 1000000 \
    --dataset porto \
    --trajsimi_measure dtw
```

### Example 2: TrjSR Model on Chengdu with Hausdorff (with meta-learning)

```bash
cd script
python TrjSR_train_script.py \
    --cluster_num 10 \
    --num_iterations 100 \
    --budget 1000000 \
    --dataset chengdu \
    --trajsimi_measure hausdorff \
    --use_meta_learning
```

### Example 3: Testing a Pre-trained Model

If you have already trained a model, you can test it:

```bash
cd script
python T3S_train_script.py \
    --cluster_num 10 \
    --num_iterations 100 \
    --budget 1000000 \
    --dataset porto \
    --trajsimi_measure dtw \
    --test true
```

### Available Models

- **T3S**: Trajectory-Trajectory-Trajectory Similarity
- **TrjSR**: Trajectory Similarity Ranking
- **TrajGAT**: Trajectory Graph Attention Network
- **TMN**: Trajectory Matching Network

### Available Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--cluster_num` | Number of clusters for trajectory grouping | Integer (e.g., 10) |
| `--num_iterations` | Number of training iterations | Integer (e.g., 100) |
| `--budget` | Computational budget | Integer (e.g., 1000000) |
| `--dataset` | Dataset to use | `porto`, `chengdu` |
| `--trajsimi_measure` | Similarity measure | `dtw`, `hausdorff`, `discret_frechet`, `sspd` |
| `--use_meta_learning` | Enable meta-learning (TrjSR only) | Flag |
| `--test` | Test mode (requires trained model) | `true` or `false` |

## Project Structure

```
GoodTP/
├── config.py                 # Configuration file
├── core/                     # Core computation modules
├── core_cpu/                 # CPU-based implementations
├── data/                     # Demo datasets
│   ├── porto_inf_trajsimi_dict_traj_all
│   └── chengdu_inf_trajsimi_dict_traj_all
├── exp/                      # Experiment outputs
├── nn/                       # Neural network models
├── output/                   # Training outputs and logs
├── script/                   # Training scripts
│   ├── T3S_train_script.py
│   ├── TrjSR_train_script.py
│   ├── TrajGAT_train_script.py
│   └── TMN_train_script.py
├── trajsimi_service/         # FastAPI trajectory similarity service
│   ├── app.py               # FastAPI application
│   ├── similarity_utils.py  # Similarity computation utilities
│   ├── run.sh               # Service startup script
│   └── requirements.txt     # Service-specific dependencies
├── utilities/                # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### Redis Connection Error

If you encounter Redis connection errors:

1. Check if Redis is running: `redis-cli ping`
2. Check Redis port (default: 6379): `redis-cli -p 6379 ping`
3. Check Redis configuration in `config.py`

### API Service Not Responding

If the trajectory similarity service doesn't respond:

1. Check if the service is running: `ps aux | grep gunicorn`
2. Check the service logs: `tail -f nohup.out`
3. Verify the port is not in use: `lsof -i :8800`
4. Restart the service:
   ```bash
   cd trajsimi_service
   pkill -f gunicorn
   bash run.sh
   ```

### CUDA Out of Memory

If you encounter CUDA memory errors:

- Reduce batch size in the training script
- Use a smaller model
- Use CPU mode by setting appropriate flags in `config.py`
