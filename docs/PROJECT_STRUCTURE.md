# DAMP Project Structure

## 📁 Directory Organization

```
DAMP/
├── 📄 run.py                          # 🚀 Main entry point
├── 📁 src/                            # 🔧 Core source code
│   ├── __init__.py                    # Package initialization
│   ├── data.py                        # Data loading and processing
│   ├── models.py                      # Neural network models
│   ├── trainer.py                     # Training utilities
│   ├── generator.py                   # Sequence generation
│   └── evaluator.py                   # Quality evaluation
├── 📁 scripts/                        # 📜 Main scripts
│   ├── main.py                        # Main training script
│   └── demo.py                        # Demo script
├── 📁 test/                           # 🧪 Test files
│   ├── run_tests.py                   # Test runner
│   ├── test_enhanced_training.py      # Enhanced training tests
│   ├── test_generation.py             # Generation tests
│   ├── test_model_comparison.py       # Model comparison tests
│   ├── test_data_loading.py           # Data loading tests
│   └── test_improved.py               # Legacy test file
├── 📁 visualization/                  # 📊 Visualization tools
│   ├── plot_training_curves.py        # Training curve plots
│   ├── advanced_plotting.py           # Advanced visualizations
│   └── training_visualization_summary.md
├── 📁 docs/                           # 📚 Documentation
│   ├── PROJECT_SUMMARY.md
│   └── roadmap.md
├── 📁 dataset/                        # 📂 Data files
│   ├── naturalAMPs_APD2024a.fasta     # AMP sequences
│   └── non-amp.fasta                  # Non-AMP sequences
├── 📁 models/                         # 💾 Trained models
├── 📁 results/                        # 📈 Generation results
├── 📁 plots/                          # 🎨 Generated plots
├── 📁 logs/                           # 📝 Training logs
├── 📄 config.py                       # ⚙️ Configuration
├── 📄 requirements.txt                # 📦 Dependencies
├── 📄 pyproject.toml                  # 📋 Project metadata
├── 📄 README.md                       # 📖 Main documentation
└── 📄 PROJECT_STRUCTURE.md            # 📋 This file
```

## 🎯 File Purposes

### Core Files
- **`run.py`**: Main entry point for the application
- **`config.py`**: Centralized configuration management
- **`requirements.txt`**: Python dependencies
- **`pyproject.toml`**: Project metadata and build configuration

### Source Code (`src/`)
- **`data.py`**: Data loading, preprocessing, and dataset classes
- **`models.py`**: Neural network architectures (GNN, Diffusion)
- **`trainer.py`**: Training classes with enhanced loss functions and early stopping
- **`generator.py`**: Sequence generation with multiple modes
- **`evaluator.py`**: Quality evaluation and metrics calculation

### Scripts (`scripts/`)
- **`main.py`**: Complete training and generation pipeline
- **`demo.py`**: Quick demonstration script

### Tests (`test/`)
- **`run_tests.py`**: Test runner for all tests
- **`test_enhanced_training.py`**: Enhanced training functionality tests
- **`test_generation.py`**: Sequence generation tests
- **`test_model_comparison.py`**: Model performance comparison tests
- **`test_data_loading.py`**: Data loading and validation tests

### Visualization (`visualization/`)
- **`plot_training_curves.py`**: Basic training curve visualization
- **`advanced_plotting.py`**: Advanced analysis and comparison plots
- **`training_visualization_summary.md`**: Visualization documentation

### Documentation (`docs/`)
- **`PROJECT_SUMMARY.md`**: Project overview and achievements
- **`roadmap.md`**: Development roadmap and future plans

### Data and Results
- **`dataset/`**: Input data files (FASTA sequences)
- **`models/`**: Saved trained models
- **`results/`**: Generation results and evaluation reports
- **`plots/`**: Generated visualization images
- **`logs/`**: Training logs and debugging information

## 🔄 File Movement History

### Moved Files
- `main.py` → `scripts/main.py`
- `demo.py` → `scripts/demo.py`
- `test_enhanced_training.py` → `test/test_enhanced_training.py`
- `test_generation.py` → `test/test_generation.py`
- `compare_models.py` → `test/test_model_comparison.py`
- `debug_load.py` → `test/test_data_loading.py`
- `plot_training_curves.py` → `visualization/plot_training_curves.py`
- `advanced_plotting.py` → `visualization/advanced_plotting.py`
- `training_visualization_summary.md` → `visualization/training_visualization_summary.md`
- `PROJECT_SUMMARY.md` → `docs/PROJECT_SUMMARY.md`
- `roadmap.md` → `docs/roadmap.md`

### New Files
- `run.py`: New main entry point
- `test/run_tests.py`: Test runner script
- `PROJECT_STRUCTURE.md`: This file

## 🚀 Usage Examples

### Running the Application
```bash
# Main application
python run.py --epochs 50 --num_sequences 100

# Demo mode
python run.py --demo

# Use existing models
python run.py --skip_training --num_sequences 50
```

### Running Tests
```bash
# Run all tests
python test/run_tests.py

# Run individual tests
python test/test_data_loading.py
python test/test_generation.py
python test/test_model_comparison.py
python test/test_enhanced_training.py
```

### Generating Visualizations
```bash
# Basic plots
python visualization/plot_training_curves.py

# Advanced analysis
python visualization/advanced_plotting.py
```

### Running Scripts
```bash
# Main training script
python scripts/main.py --epochs 100

# Demo script
python scripts/demo.py
```

## 📋 Naming Conventions

### Files
- **Test files**: `test_*.py` (e.g., `test_data_loading.py`)
- **Script files**: Descriptive names (e.g., `main.py`, `demo.py`)
- **Configuration**: `config.py`
- **Documentation**: `*.md` files

### Directories
- **Source code**: `src/`
- **Tests**: `test/`
- **Scripts**: `scripts/`
- **Visualization**: `visualization/`
- **Documentation**: `docs/`
- **Data**: `dataset/`
- **Output**: `models/`, `results/`, `plots/`, `logs/`

## 🔧 Development Workflow

1. **Add new features** → `src/` modules
2. **Create tests** → `test/` directory
3. **Add visualizations** → `visualization/` directory
4. **Update documentation** → `docs/` directory
5. **Run tests** → `python test/run_tests.py`
6. **Generate plots** → `python visualization/advanced_plotting.py`

## 📊 Project Statistics

- **Total files**: 25+
- **Source files**: 6 core modules
- **Test files**: 5 test scripts
- **Visualization files**: 3 plotting scripts
- **Documentation files**: 4 markdown files
- **Generated plots**: 11 visualization images

---

**Last Updated**: 2024-08-24  
**Organization Status**: ✅ Complete 