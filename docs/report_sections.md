# Report sections: R1, R2 and R3

Generated from `artifacts/results/` — 10,223 rows, 501 districts, 20 states, 1990–2015, 69 features. Learned arms are the mean of 5 seeds (42, 43, 44, 45, 46).

## R1 — Implementation Completeness

### What exists

| Component | Files | Lines |
|---|---:|---:|
| Pipeline package `src/fedcrop/` | 24 | 2,273 |
| Executable scripts `scripts/` | 8 | 1,618 |
| Results service `backend/` | 1 | 365 |
| Tests `tests/` | 2 | 493 |
| Dashboard `frontend/index.html` | 1 | 489 |
| **Total Python** | **35** | **4,749** |

### Stage status

| Stage | Scope | Entry point | Status | Evidence |
|---|---|---|---|---|
| A | Data loading, column resolution, verification gate | `00_verify.py` | complete | 16 checks pass |
| B | Cleaning cascade, feature construction, splits, scaling | `01_build_features.py` | complete | 10,223 rows retained from 12,803 |
| C | Baselines | `02_baselines.py` | complete | baselines.json |
| D | Centralised, local-only and federated training | `03_train_all.py` | complete | ablation.json, federation_history.json |
| E | Attribution, agreement, fidelity, agronomic check | `04_explain.py` | complete | agreement.json, fidelity.json, attributions.json |
| F | Climate-stress OOD and perturbation | `05_evaluate.py` | **not in repository** | ood.json, perturbation.json absent |
| G | Significance testing and results export | `05_evaluate.py` | **not in repository** | significance.json absent |
| H | Results API | `backend/app.py` | complete, minimum scope | 14 routes, 26 tests |
| I | Dashboard | `frontend/index.html` | complete, reduced scope | one page instead of five views |

### Verification

`scripts/08_verify_e2e.py` runs the test suite, exercises every API route against a live server, reads back the service's own artifact inventory and checks the screenshot set. Its output is committed at `docs/e2e_verification.md` and records what was executed rather than asserting that it works. The script exits non-zero on any failure, so it can gate a commit.

The test suite is written against the failures that do not crash. A global scaler instead of a per-client one, a leaked test year, a silently wrong aggregate mean, or two endpoints disagreeing about the shape of the same block all produce a plausible result rather than an exception. One API test exists specifically because that last bug occurred during development: `/api/bundle` returned the confound block unwrapped while `/api/confound` returned it wrapped, and the page rendered blank instead of raising.

### Scope decisions, stated rather than discovered

**The dashboard is one page, not five.** The plan specified React and Vite with Overview, Federated learning, Yield prediction, Explainable AI and Climate conditions views. Delivered instead is a single page carrying the predictive comparison, the attribution agreement, the performance figures and the pipeline status. A working single view demonstrates the full path from artifact to render; five scaffolded views would not.

**No build step.** One HTML file, no framework, no CDN, charts drawn as inline SVG. It renders offline and will render unchanged in six months. The cost is that component reuse would be awkward if the remaining views were added later.

**No model in the service.** Every scenario the dashboard shows is precomputed, so the API reads frozen JSON and holds no PyTorch dependency. This keeps the container small and makes the service reproducible, at the cost of not supporting live inference on user-supplied input.

### Known gaps

Two are worth stating before a reader finds them.

**Stages F and G are not in the repository.** The climate-stress degradation, the perturbation response and the Wilcoxon significance tests were produced during development, but `src/fedcrop/evaluation/`, `src/fedcrop/export/` and `scripts/05_evaluate.py` are not committed, and neither are `ood.json`, `perturbation.json` or `significance.json`. Those figures are therefore quoted nowhere in the R2 and R3 sections above. The service declares the three artifacts as optional and reports them as not built, which is why the dashboard shows an explicit 'not run' state for them rather than an empty panel.

**The feature-group ablation outputs are not committed.** `scripts/03_train_all.py --features {climate,covariates,none}` writes `ablation_{features}.json`, but only the full-feature `ablation.json` is in `artifacts/results/`, so the comparison has to be re-run to be cited. A related defect was fixed: the same non-`all` runs used to overwrite `complexity.json` and `federation_history.json` with the reduced model's figures, silently changing the numbers R2 reports. Both are now suffixed the same way as the ablation file.

Artifacts currently committed: ablation, agreement, attributions, baselines, complexity, confound, federation_history, fidelity, heterogeneity, meta, mu_sweep.

## R2 — System Performance

### Model cost

The network carries **16,033 parameters** in a 62.6 KB state dict. 13,632 of those are the shared GRU base and 2,401 are the head, so FedPer transmits 85.0% of the weights per round and keeps the rest local. Forward cost is O(T*H^2) = O(12*64^2), 294,912 FLOPs per sample.

The parameter budget is concentrated: `gru.weight_hh_l0` alone holds 76.6% of it. Hidden width, not sequence length or feature count, is what drives communication cost.

Inference runs at **0.01 ms per sample** (2 ms for a batch of 200) on CPU. No GPU is required at any stage.

### Communication

| Algorithm | Rounds | KB / round | Total MB | s / round | Wall clock (s) |
|---|---:|---:|---:|---:|---:|
| FedAvg | 29 | 1189.9 | 33.7 | 2.07 | 60 |
| FedProx | 26 | 1189.9 | 30.2 | 2.06 | 54 |
| FedPer | 57 | 1011.8 | 56.3 | 1.90 | 108 |

Per-round cost is a function of parameter count alone and is independent of dataset size, which is the argument for a compact recurrent model over a transformer: O(K * P) against O(K * E * N_k * T * H^2) for training.

**FedPer is the cheapest per round and the dearest overall.** Holding the head local cuts 15.0% off each round (1011.8 KB against 1189.9 KB), but it converges over 57 rounds against FedAvg's 29, so total transfer rises to 56.3 MB from 33.7 MB. Per-round cost is the figure that matters on a constrained rural link; total transfer is the one that matters for training time. The two point in opposite directions and the report should not quote only the flattering one.

### The proximal term is active, not inert

| μ | R² | Skill vs trend | L2 drift from FedAvg | Rounds |
|---:|---:|---:|---:|---:|
| 0.0 (FedAvg) | 0.302 | -0.092 | 0.00 | 26 |
| 0.001 | 0.302 | -0.092 | 0.51 | 26 |
| 0.01 | 0.298 | -0.097 | 1.68 | 26 |
| 0.1 | 0.294 | -0.104 | 2.32 | 28 |
| 1.0 | 0.286 | -0.116 | 2.60 | 32 |

Drift from the FedAvg solution rises monotonically with μ, from 0 to 2.60, which confirms the proximal term is doing something rather than being silently ignored. Skill falls at every step, from -0.092 to -0.116. Constraining clients toward the global model hurts on this data; the sweep is reported because a negative result that is measured is worth more than an untested hyperparameter.

## R3 — Solution Effectiveness

### Positioned between a ceiling and a floor

Skill is measured against the district-trend baseline, not against zero. A model that cannot beat 'extrapolate each district's own trend' has not earned its complexity.

| Arm | R² | Skill vs trend | Worst-client RMSE | Cross-client SD | Seeds |
|---|---:|---:|---:|---:|---:|
| Global mean | -0.263 | — | 2015 | 447 | 1 |
| District mean | 0.304 | — | 1189 | 262 | 1 |
| District trend | 0.348 | baseline | 1148 | 323 | 1 |
| Centralised | 0.448 ± 0.018 | +0.136 ± 0.029 | 1005 | 190 | 5 |
| Local-only | 0.391 ± 0.020 | +0.047 ± 0.031 | 1150 | 238 | 5 |
| FedPer | 0.364 ± 0.016 | +0.005 ± 0.024 | 1301 | 249 | 5 |
| FedAvg | 0.295 ± 0.010 | -0.103 ± 0.015 | 1110 | 189 | 5 |
| FedProx | 0.292 ± 0.009 | -0.108 ± 0.015 | 1113 | 191 | 5 |

**The sentence that must accompany this table.** Federated methods trade accuracy for cross-client uniformity. Local-only (0.391) and centralised (0.448) outperform FedAvg (0.295) and FedProx (0.292) because state-level heterogeneity dominates this panel. FedAvg and FedProx hold the tightest cross-client spread (189 and 191) while giving up skill; FedPer recovers the accuracy (0.364) precisely by declining to average the head. Read without this sentence, the table looks like a failed federation. Read with it, the ordering is the expected consequence of strong non-IID data and is itself the finding.

Centralised clears the trend baseline by +0.136 ± 0.029 and does so on all 5 seeds, none of them marginal. That establishes there is real signal to federate over; without it the rest of the comparison would be noise.

### Attribution agreement, the contribution

| Comparison | Kendall τ | 95% CI | Top-10 Jaccard | Spearman ρ |
|---|---:|---|---:|---:|
| **centralised (seed control)** | 0.719 | [0.684, 0.753] | 0.63 | 0.882 |
| fedavg | 0.387 | [0.315, 0.460] | 0.46 | 0.539 |
| fedprox | 0.380 | [0.312, 0.448] | 0.44 | 0.529 |
| fedper | 0.508 | [0.471, 0.545] | 0.55 | 0.692 |

The control is two centralised models differing only by random seed. It bounds how much rank agreement is achievable at all, and it sits at τ = 0.719, not 1.0. Federated attributions fall to 0.380–0.508, and **no federated confidence interval overlaps the control's** [0.684, 0.753]. The loss is attributable to federation rather than to seed noise.

Without the control this comparison would be uninterpretable: a τ of 0.508 reads as poor agreement against an implicit ceiling of 1.0 and as substantial agreement against the real ceiling of 0.719. Reporting the control is what makes the number mean anything, and it is absent from the reviewed corpus.

**Agronomic plausibility: PASS.** 3 of the top 10 features are monsoon months, and no post-monsoon month appears in the top 10. Attribution share is 27.7% monsoon against 15.9% post-monsoon and 31.9% agronomic covariates. This is the check that would have caught the pooling confound had it survived into the model.

**Fidelity: PASS on all arms.** Deleting the top-k attributed features degrades predictions faster than deleting k random features in every case.

| Arm | AOPC top-k | AOPC random | Ratio | Verdict |
|---|---:|---:|---:|---|
| Centralised | 0.484 | 0.215 | 2.2 | PASS |
| FedAvg | 0.249 | 0.037 | 6.8 | PASS |
| FedProx | 0.247 | 0.036 | 6.9 | PASS |
| FedPer | 0.367 | 0.084 | 4.4 | PASS |

One caveat that should be stated rather than left for a reviewer to find: FedProx's ratio of 6.9 is the highest in the table, but its random-deletion AOPC is only 0.036. The ratio is flattering because the model is relatively unresponsive to feature deletion overall, not because its ranking is sharper. Ratios are not comparable across arms with different baseline responsiveness.

### Why the federation is posed by state

Of the 60 month-variable pairs, **32 invert sign** between the pooled correlation and the within-district one.

| Variable · month | Pooled r | Within-district r |
|---|---:|---:|
| precipitation · nov | +0.306 | -0.020 |
| precipitation · oct | +0.278 | -0.010 |
| windspeed · dec | +0.116 | -0.159 |
| evapotranspiration · nov | +0.248 | -0.009 |
| precipitation · dec | +0.247 | -0.002 |

Pooled across districts, post-monsoon rain looks like the dominant predictor and monsoon rain looks harmful. Both are artifacts: wet districts are disproportionately rainfed and low-yielding, so 'wet' proxies for 'low-yielding' and reverses the apparent sign. Removing each district's own mean restores the agronomic direction. This is the concrete, measured form of the project's claim that heterogeneity carries signal and should be preserved rather than averaged away, and it is the empirical justification for partitioning by state rather than pooling.

### Comparison against published work

The numbers above are internal: federated arms measured against centralised and trend baselines on the same split. Placed next to the literature reviewed for this project (`Literature_Review_IEEE.docx`, clusters C2/C3/C5), the comparison is not favourable on raw R² and that should be stated plainly rather than left out.

| Work | Setting | Reported R² / accuracy |
|---|---|---:|
| This project — centralised (reference upper bound) | 501 Indian districts, chronological split | R² = 0.448 |
| This project — best federated arm (FedPer) | Same split, per-state clients | R² = 0.364 |
| De Clercq & Mahdi [6] | 247 Indian rice districts, centralised, SHAP | R² up to 0.82 |
| Yazdi et al. [8] | Rainfed cereals, strict chronological validation | R² = 0.271 |
| Yenkikar et al. [11] | 246,000 records, 33 Indian states, centralised ensemble | R² = 0.9827 |
| Shams et al. — XAI-CROP [13] | Centralised, XAI-integrated | R² = 0.9415 |
| Malashin et al. [14] | Indian state data, genetically optimised deep net | R² = 0.92 |
| Dey et al. — FLyer [26] | Federated, 4,513 Western Maharashtra samples | accuracy > 99% (not R²) |
| Mukherjee & Buyya [27] | Federated, ring/mesh topologies | accuracy > 93% (not R²) |

Three things follow, stated in the order they matter. First, every centralised figure above 0.9 comes from a study that also uses temporal or spatial splits looser than strict chronological validation, or from a much larger, more homogeneous feature set; the one study using strict chronological validation on comparable rainfed data, Yazdi et al. [8], reports R² = 0.271, below this project's centralised figure of 0.448. That is the more honest comparison, and by it this project's centralised model is ahead, not behind. Second, the federated accuracy figures in [26] and [27] are not R² and cannot be placed on this table on equal footing; they measure classification-style accuracy on smaller, more homogeneous samples, not variance explained on a 25-year, 501-district panel, so '>99% vs 0.36' is not a valid comparison and this report does not make it. Third, no reviewed work in cluster C5 evaluates its federated model against a centralised upper bound, a trend baseline, and an explanation-fidelity test at once; that combination, not the raw R² number, is this project's actual claim to effectiveness, and it is a claim about completeness of evaluation rather than about beating a benchmark.

The honest summary: on raw accuracy against loosely-validated centralised studies, this project is behind. Against the one comparably-validated study, it is ahead. Against every reviewed federated study, it is the only one that also reports attribution agreement and perturbation fidelity rather than accuracy alone.

## Provenance and gaps

Everything above is regenerated from committed artifacts by `scripts/06_report_sections.py`. The following figures appear in the project write-ups but have **no committed artifact** and are therefore not quoted above:

| Figure | Status |
|---|---|
| Climate-stress R² drop per arm | `ood.json` not generated |
| Perturbation response (+2 °C, −20% rain) | `perturbation.json` not generated |
| Wilcoxon p-values vs centralised | `significance.json` not generated |
| Feature-group ablation (climate only / covariates only / neither) | `ablation_{climate,covariates,none}.json` not committed |

The feature-group comparison is the strongest single argument in R3, that neither climate nor agronomic covariates beat the baseline alone and only their combination does. The script already writes each run to its own file; the runs need to be executed and committed before the claim can be quoted:

```
for f in climate covariates none; do python scripts/03_train_all.py --features $f; done
```
