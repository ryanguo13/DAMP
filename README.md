# DAMP: Diffusion-based Antimicrobial Peptide Generator

A deep learning system for generating novel antimicrobial peptides using diffusion models and graph neural networks.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DAMP

# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Basic Usage

```bash
# Run the main application
python run.py --epochs 50 --num_sequences 100

# Demo mode (quick training)
python run.py --demo

# Use existing models
python run.py --skip_training --num_sequences 50
```

## 📁 Project Structure

```
DAMP/
├── run.py                          # Main entry point
├── src/                            # Core source code
│   ├── __init__.py
│   ├── data.py                     # Data loading and processing
│   ├── models.py                   # Neural network models
│   ├── trainer.py                  # Training utilities
│   ├── generator.py                # Sequence generation
│   └── evaluator.py                # Quality evaluation
├── scripts/                        # Main scripts
│   ├── main.py                     # Main training script
│   └── demo.py                     # Demo script
├── test/                           # Test files
│   ├── test_enhanced_training.py   # Enhanced training tests
│   ├── test_generation.py          # Generation tests
│   ├── test_model_comparison.py    # Model comparison tests
│   ├── test_data_loading.py        # Data loading tests
│   └── test_improved.py            # Legacy test file
├── visualization/                  # Visualization tools
│   ├── plot_training_curves.py     # Training curve plots
│   ├── advanced_plotting.py        # Advanced visualizations
│   └── training_visualization_summary.md
├── docs/                           # Documentation
│   ├── PROJECT_SUMMARY.md
│   └── roadmap.md
├── dataset/                        # Data files
│   ├── naturalAMPs_APD2024a.fasta  # AMP sequences
│   └── non-amp.fasta              # Non-AMP sequences
├── models/                         # Trained models
├── results/                        # Generation results
├── plots/                          # Generated plots
├── logs/                           # Training logs
├── config.py                       # Configuration
├── requirements.txt                # Dependencies
└── pyproject.toml                  # Project metadata
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python test/test_enhanced_training.py
python test/test_generation.py
python test/test_model_comparison.py
python test/test_data_loading.py
```

### Data Loading Test

```bash
python test/test_data_loading.py
```

### Model Comparison

```bash
python test/test_model_comparison.py
```

## 📊 Visualization

### Generate Training Plots

```bash
# Basic training curves
python visualization/plot_training_curves.py

# Advanced visualizations
python visualization/advanced_plotting.py
```

### Available Visualizations

- Training loss curves
- Model comparison heatmaps
- Performance radar charts
- Training analysis dashboard
- Improvement analysis plots

## 🔧 Configuration

### Model Configuration

```python
# In config.py
ModelConfig(
    embed_dim=64,
    hidden_dim=128,
    num_layers=3,
    noise_steps=20
)
```

### Training Configuration

```python
TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=10
)
```

## 🎯 Features

### Enhanced Training

- **FocalLoss**: Handles class imbalance
- **Label Smoothing**: Improves generalization
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates

### Model Architecture

- **GNN Scorer**: Graph Neural Network for AMP prediction
- **Diffusion Model**: Denoising diffusion for sequence generation
- **Enhanced Loss Functions**: Better training stability

### Quality Evaluation

- **Diversity Metrics**: Sequence diversity analysis
- **AMP Potential**: Antimicrobial activity prediction
- **Validity Checks**: Sequence validity verification
- **Comprehensive Reports**: Detailed quality assessment

## 📈 Performance

### GNN Model Improvements

- Validation loss: -25.1% (0.2706 → 0.2026)
- Validation accuracy: +1.9% (92.35% → 94.10%)
- Training efficiency: +7.7% (13 → 12 epochs)

### Dataset Balance

- **Before**: 3224 AMP + 2 non-AMP sequences
- **After**: 3224 AMP + 3050 non-AMP sequences
- **Improvement**: Balanced dataset for realistic evaluation

## 🙏 Acknowledgments

- APD2024 dataset for AMP sequences
- PyTorch for deep learning framework
- Biopython for sequence processing
- Matplotlib/Seaborn for visualizations
