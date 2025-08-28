## Phase 1: Model prep

## Phase 2: Data Preparation with ESM3 Embeddings (1-2 weeks)

* **Objectives** : Augment your dataset with ESM3 features for GNN and diffusion inputs.
* **Steps** :

1. Update src/data.py: Add a function to compute ESM3 embeddings for sequences.
   python

   `<div><div><div><span><span>def</span><span></span><span>get_esm3_embeddings</span><span>(</span><span>sequences, tokenizer, model, max_len=</span><span>50</span><span>, batch_size=</span><span>32</span><span>):</span><span> </span></span></div><div><span>    embeddings = [] </span></div><div><span><span></span><span>for</span><span> i </span><span>in</span><span></span><span>range</span><span>(</span><span>0</span><span>, </span><span>len</span><span>(sequences), batch_size): </span></span></div><div><span>        batch = sequences[i:i+batch_size] </span></div><div><span><span>        inputs = tokenizer(batch, return_tensors=</span><span>"pt"</span><span>, padding=</span><span>True</span><span>, truncation=</span><span>True</span><span>, max_length=max_len).to(model.device) </span></span></div><div><span><span></span><span>with</span><span> torch.no_grad(): </span></span></div><div><span><span>            outputs = model(**inputs, output_hidden_states=</span><span>True</span><span>) </span></span></div><div><span><span>        emb = outputs.hidden_states[-</span><span>1</span><span>].mean(dim=</span><span>1</span><span>)  </span><span># Or per-residue: outputs.hidden_states[-1]</span><span> </span></span></div><div><span>        embeddings.append(emb.cpu()) </span></div><div><span><span></span><span>return</span><span> torch.cat(embeddings)</span></span></div></div></div>`

   * For per-residue embeddings: Use them as node features (shape: [seq_len, embed_dim]).
   * For sequence-level: Use pooled embeddings for classification/regression.
2. Precompute and cache: Run on your dataset/amp.fasta and non_amp.fasta. Save as .pt files (e.g., dataset/esm3_embeddings.pt) to avoid recomputing.
3. Update PeptideDataset: Modify __getitem__ to include ESM3 embeddings (e.g., as additional node attributes in the graph).

   * For GNN: Concat ESM3 per-residue embeddings with existing features (physico-chemical, secondary structure).
4. Balance integration: If embeddings are high-dimensional, add a projection layer (e.g., Linear(embed_dim, hidden_dim)) to reduce to match your GNN input size.

* **Milestones** : Dataset loader now returns ESM3-augmented features; verify with a dataloader iteration.
* **Dependencies** : Phase 1 setup; your existing data loading.
* **Risks** : Long computation time for large datasets; process in batches and use multiple GPUs if available.

## Phase 3: Integrate ESM3 into GNN Scorer (1-2 weeks)

* **Objectives** : Enhance GNN with ESM3 embeddings for better AMP scoring (classification/MIC regression).
* **Steps** :

1. Update src/models.py GNN class:

   * Replace or augment self.embedding (current: nn.Embedding) with ESM3 features.

   python

   `<div><div><div><span><span>class</span><span></span><span>GNNScorer</span><span>(</span><span>nn.Module</span><span>):</span><span> </span></span></div><div><span><span></span><span>def</span><span></span><span>__init__</span><span>(</span><span>self, esm_embed_dim=</span><span>768</span><span>, hidden_dim=</span><span>64</span><span>):</span><span></span><span># Adjust based on ESM3 variant</span><span> </span></span></div><div><span><span></span><span>super</span><span>().__init__() </span></span></div><div><span><span>        self.proj = nn.Linear(esm_embed_dim, hidden_dim)  </span><span># Project ESM3 to hidden_dim</span><span> </span></span></div><div><span>        self.conv1 = GCNLayer(hidden_dim, hidden_dim) </span></div><div><span><span></span><span># ... (rest as before)</span><span> </span></span></div><div><span> </span></div><div><span><span></span><span>def</span><span></span><span>forward</span><span>(</span><span>self, esm_emb, adj, mask</span><span>):</span><span></span><span># Input now includes precomputed esm_emb [batch, seq_len, esm_dim]</span><span> </span></span></div><div><span>        x = F.relu(self.proj(esm_emb)) </span></div><div><span><span></span><span># Proceed with convolutions as before</span></span></div></div></div>`

   * If using per-residue: Pass esm_emb directly; for sequence-level, pool early.
2. Modify training in src/trainer.py: Load ESM3 model once and pass embeddings to GNN.
3. Fine-tune: Retrain GNN with ESM3 features. Add contrastive loss if needed to align ESM3 with your task.
4. Evaluate: Compare metrics (AUC-ROC, MAE) before/after integration using your src/evaluator.py.

* **Milestones** : GNN accuracy improved (target: +5-10% on validation); logged in logs/.
* **Dependencies** : Phase 2 data.
* **Risks** : Overfitting due to rich features; use dropout (0.1-0.3) and regularization.

## Phase 4: Integrate ESM3 into Diffusion Generator (1-2 weeks)

* **Objectives** : Condition diffusion on ESM3 for more biologically plausible generations.
* **Steps** :

1. Update src/models.py Denoiser:
   python

   `<div><div><div><span><span>class</span><span></span><span>Denoiser</span><span>(</span><span>nn.Module</span><span>):</span><span> </span></span></div><div><span><span></span><span>def</span><span></span><span>__init__</span><span>(</span><span>self, esm_embed_dim=</span><span>768</span><span>, hidden_dim=</span><span>128</span><span>):</span><span> </span></span></div><div><span><span></span><span>super</span><span>().__init__() </span></span></div><div><span><span>        self.embedding = nn.Embedding(num_aa + </span><span>1</span><span>, embed_dim)  </span><span># Keep for noised input</span><span> </span></span></div><div><span><span>        self.esm_proj = nn.Linear(esm_embed_dim, hidden_dim)  </span><span># Project ESM3</span><span> </span></span></div><div><span><span>        self.fc1 = nn.Linear(embed_dim + </span><span>1</span><span> + hidden_dim, hidden_dim)  </span><span># Concat noise emb + t + esm</span><span> </span></span></div><div><span> </span></div><div><span><span></span><span>def</span><span></span><span>forward</span><span>(</span><span>self, noised, t, mask, esm_emb</span><span>):</span><span></span><span># Add esm_emb input</span><span> </span></span></div><div><span>        x = self.embedding(noised) </span></div><div><span><span>        esm_proj = self.esm_proj(esm_emb)  </span><span># [batch, seq_len, hidden_dim]</span><span> </span></span></div><div><span><span>        t_exp = t.unsqueeze(</span><span>1</span><span>).repeat(</span><span>1</span><span>, x.size(</span><span>1</span><span>), </span><span>1</span><span>) </span></span></div><div><span><span>        x = torch.cat([x, t_exp, esm_proj], dim=-</span><span>1</span><span>) </span></span></div><div><span><span></span><span># ... (rest as before)</span></span></div></div></div>`

   * During generation: At each denoising step, compute ESM3 embeddings on the current noised sequence and feed in.
2. Update src/generator.py: In generate_sequence, insert ESM3 embedding computation inside the loop.
3. RL Loop: If using rewards (GNN scores), weigh with ESM3-predicted structure quality (e.g., ESM3's structure tokens for stability).
4. Train: Update src/trainer.py to precompute ESM3 on clean sequences during diffusion training.

* **Milestones** : Generated sequences show higher AMP scores and diversity (check via src/evaluator.py).
* **Dependencies** : Phase 3 GNN (for rewards).
* **Risks** : Increased inference time; optimize by caching or approximating ESM3 calls.

## Phase 5: End-to-End Testing, Optimization, and Validation (1-2 weeks)

* **Objectives** : Validate the integrated pipeline and refine.
* **Steps** :

1. Update scripts/main.py and demo.py: Add flags for ESM3 usage (e.g., --use_esm3).
2. Benchmark: Run full pipeline; compare generations (pre/post-ESM3) on metrics like novelty, AMP potential.
3. Optimization: Hyperparameter tune (e.g., embedding fusion weights) using your enhanced training features (FocalLoss, etc.).
4. Validation: Use external tools (e.g., AlphaFold via ESM3 outputs) for structure checks; select top candidates for in silico MIC prediction.
5. Documentation: Update docs/roadmap.md and README with ESM3 section; add examples in test/.

* **Milestones** : Pipeline runs end-to-end; improved performance documented in results/.
* **Dependencies** : All prior phases.
* **Risks** : Compatibility issues; test incrementally.

## Additional Considerations

* **Compute Scaling** : ESM3 inference is heavy; use distributed training if scaling up.
* **Alternatives** : If ESM3 is too large, start with smaller ESM variants (e.g., ESM2).
* **Future Extensions** : Fine-tune ESM3 on your AMP data using PEFT; integrate ESM3's generative capabilities for direct peptide design.
* **Monitoring** : Track in logs/ and plots/; aim for iterative improvements.

This integration should significantly boost your model's biological accuracy. Start with Phase 1 and prototype in a Jupyter notebook for quick testing. If issues arise (e.g., with specific code snippets), provide more details from your repo for refinements.
