# Diffusion-Driven Antimicrobial Peptide Engineering with GNN

## Project Overview

This project adapts the concepts with a Diffusion model for sequence generation, while retaining Graph Neural Networks (GNN) for structural scoring. The pipeline integrates reinforcement learning (RL) elements implicitly through iterative generation and scoring. The goal is to engineer novel Antimicrobial Peptides (AMPs) efficiently using motif-guided strategies, combinatorial templates, and AI-driven generation.

Datasets:

* **AMPs** : Sourced from the Antimicrobial Peptide Database (APD).
* **Non-AMPs** : Sourced from UniProt, filtered for non-antimicrobial sequences (e.g., short peptides without AMP annotations).

The pipeline focuses on generating peptides (e.g., length ≤ 50), scoring them for AMP potential (classification and regression, e.g., MIC prediction), and iterating via RL-guided feedback.

## Roadmap

The project follows a phased approach, building from data preparation to deployment and validation. Each phase includes milestones, dependencies, and estimated timelines (assuming a small team; adjust based on resources).

### Phase 1: Data Collection and Preparation (1-2 weeks)

* **Objectives** : Gather and preprocess datasets for training.
* **Steps** :

1. Download AMP sequences from APD (e.g., via FASTA export from their database website).
2. Query UniProt for non-AMP peptides (e.g., using filters for short sequences without "antimicrobial" keywords or annotations; download in FASTA format).
3. Filter sequences: Limit to lengths ≤ 50, remove duplicates, balance classes (AMPs vs. non-AMPs).
4. Augment with features: Compute physico-chemical properties (e.g., charge, hydrophobicity) and secondary structures (e.g., via Chou-Fasman or AlphaFold predictions).

* **Milestones** : Curated datasets (amp.fasta, non_amp.fasta) with labels.
* **Dependencies** : Access to APD and UniProt APIs/websites; Biopython for parsing.
* **Risks** : Data imbalance; mitigate by oversampling AMPs or undersampling non-AMPs.

### Phase 2: Model Development - GNN Scorer (2-3 weeks)

* **Objectives** : Build and train a GNN for peptide scoring (AMP classification and MIC regression).
* **Steps** :

1. Represent peptides as graphs: Nodes as residues (with embeddings for amino acids, secondary structure attributes), edges for sequential connections (including self-loops).
2. Implement GNN layers (e.g., Graph Convolutional Networks with attention/message passing).
3. Train on labeled data: Use BCE loss for classification; add contrastive learning and pseudo-label distillation for robustness.
4. Integrate pre-filters: Random Forest (RF) for quick physico-chemical and secondary structure checks.
5. Evaluate: Metrics like accuracy, AUC-ROC for classification; MAE/RMSE for regression.

* **Milestones** : Trained GNN model with >85% validation accuracy on AMP classification.
* **Dependencies** : Phase 1 data; PyTorch for implementation.
* **Risks** : Overfitting on small datasets; use cross-validation and regularization.

### Phase 3: Model Development - Diffusion Generator (2-3 weeks)

* **Objectives** : Develop a diffusion-based generator for novel peptide sequences.
* **Steps** :

1. Warm-start with initial pool: Feature-guided (e.g., motifs from known AMPs) and combinatorial templates.
2. Implement discrete diffusion: Noise sequences by corrupting amino acids (e.g., random replacement based on noise level t).
3. Build denoiser: MLP or transformer-based model to predict original from noised sequences, conditioned on t.
4. Train on AMP sequences: Use cross-entropy loss, focusing on variable-length peptides with padding/masking.
5. Integrate RL loop: Use GNN scores as rewards; update generator via policy gradients (e.g., ∇θJ(θ) = E[R ∇ log π(seq)]).

* **Milestones** : Generator producing diverse, valid peptides (e.g., 100 samples with high novelty).
* **Dependencies** : Phase 2 GNN for rewards.
* **Risks** : Mode collapse in generation; incorporate diversity metrics (e.g., sequence similarity).

### Phase 4: Integrated Pipeline and Optimization (3-4 weeks)

* **Objectives** : Close the loop with RL and multi-objective optimization.
* **Steps** :

1. Pipeline assembly: Warm-start pool → RF pre-filter → Diffusion generation → GNN scoring → RL feedback.
2. Pareto selection: Rank generated peptides on activity (AMP score), toxicity, stability (e.g., using additional predictors).
3. Iterative refinement: Fine-tune models with synthetic validation data.
4. Hyperparameter tuning: Use grid search or Bayesian optimization for diffusion steps, GNN layers, etc.
5. Benchmark: Compare against baselines (e.g., original PFT method, random generation).

* **Milestones** : End-to-end pipeline generating top-10 candidates with high scores.
* **Dependencies** : All prior phases.
* **Risks** : Computational cost; optimize with GPU acceleration and batching.

### Phase 5: Validation and Deployment (2-4 weeks)

* **Objectives** : Validate candidates and prepare for real-world use.
* **Steps** :

1. In silico validation: Predict MIC, toxicity (e.g., via external tools like hemolytic predictors).
2. Synthesis & wet-lab feedback: Select top candidates for synthesis; incorporate MIC validation into RL/GNN retraining.
3. Deployment: Package as a script or web app (e.g., Streamlit for UI).
4. Documentation: Update README, add examples.
5. Future work: Scale to larger datasets, integrate 3D structures from AlphaFold.

* **Milestones** : Validated peptides; open-source repo.
* **Dependencies** : Access to lab resources for validation.
* **Risks** : Discrepancy between in silico and in vitro; start with known positives.

### Overall Timeline

* Total: 10-16 weeks.
* Key Checkpoints: End of each phase with demos (e.g., generated samples).
* Scalability: Use cloud resources (e.g., Google Colab) for training.

## Technology Stack

### Programming Languages & Frameworks

* **Python 3.12+** : Core language for all scripts.
* **PyTorch 2.0+** : For GNN and Diffusion models (tensors, autograd, neural networks).

### Libraries & Tools

* **Data Handling** :
* Biopython: Sequence parsing (FASTA loading, manipulation).
* NumPy: Array operations, adjacency matrices.
* Pandas: Dataframe management for features/labels (optional for analysis).
* **Machine Learning** :
* Torch Geometric (or custom): For GNN implementations (if extending beyond basic GCN).
* Scikit-learn: RF pre-filter, metrics (e.g., AUC, confusion matrix).
* **Feature Extraction** :
* Custom scripts: Physico-chemical features (e.g., via Peptides library if available, or manual calc).
* AlphaFold (via API or local): For 3D graphs and secondary structures.
* **Utilities** :
* Matplotlib/Seaborn: Visualization (e.g., loss curves, peptide distributions).
*
