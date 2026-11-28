# Long-Training Study Results Summary (`snap_v3`, `2026112501`)

## 1. Experimental Setup & Audit Integrity
- **Dataset snapshot:** `data_v2_2026112501/snapshots/snap_v3` (SHA-256 `04cccaeab8d005b2ca8128d75e96c9e41dde55005076e78439cb892f95548870`), built from audited base + Waves 1–2 under `terminal_aware_v1` and `research_rules_v2_royal_conversion` with `0` censoring (0 unresolved continuations).
- **No-trimming & coverage invariants:** `max_seq_len=32768`, `require_zero_dropped_demos=True`. All 8 test evaluations achieved `100.0%` feasible and common root coverage (`1,773 / 1,773` test roots across all 14 evaluation settings: `standard`, 4 `familiar_adaptation` rules, and 9 `heldout_transfer` variants; `common_cohort == feasible_cohort` to machine precision) and `100.0%` foreign-control match coverage (`1,773 / 1,773`, `0` failures across all `K` and 5 demo draws).
- **Models & hardware constraint:** Executed on NVIDIA H100 80GB HBM3 GPUs 1–4.
  - **Track 1 (`qwen`):** `Qwen2.5-7B-Instruct` BF16 base + FP32 LoRA/head (`731,651` trainable of `7,616,348,163` total parameters; preflight `3,128.84 tok/s`, `17,806 MB` reserved VRAM).
  - **Track 2 (`scratch`):** 494M from-scratch Qwen2-style classifier (`494,150,147` trainable FP32 parameters = `136,249,344` embedding + `357,900,803` other; preflight `4,889.25 tok/s`, `10,030 MB` reserved VRAM).

## 2. Training & Full-Validation Checkpoint Selection
- **Warmup (6/6 runs completed `1,000/1,000` steps on `snap_v2`, 0 retries):**
  - Qwen val-panel best NLL: seed 42 `1.01109 @ step 900`, seed 123 `1.01690 @ step 1000`, seed 456 `1.01238 @ step 1000`.
  - Scratch val-panel best NLL: seed 42 `1.06147 @ step 400`, seed 123 `1.08136 @ step 900`, seed 456 `1.08398 @ step 600`.
- **Main runs (8/8 slots completed `10,000/10,000` steps on `snap_v3`, `80,000` episodes each, 0 retries) & frozen full-validation selection (`val_select_slot*.json`):**

| Slot | Arch | Regime | Seed | Tokens Seen | Best Val-Panel NLL | Frozen Val Selected Ckpt | Val Macro NLL | Val Macro Brier Vec |
|---|---|---|---:|---:|---:|---|---:|---:|
| `slot0` | `qwen` | `familiar` | 42 | 86,082,946 | `0.93162 @ 10000` | `step_10000` | `0.94082` | `0.05485` |
| `slot1` | `qwen` | `familiar` | 123 | 86,402,011 | `0.94568 @ 9500` | `step_10000` | `0.95091` | `0.06133` |
| `slot2` | `qwen` | `familiar` | 456 | 86,465,952 | `0.94257 @ 9500` | `step_09500` | `0.95066` | `0.06187` |
| `slot3` | `scratch` | `familiar` | 42 | 86,082,946 | `1.01149 @ 8500` | `step_08500` | `1.00133` | `0.09365` |
| `slot4` | `scratch` | `familiar` | 123 | 86,402,011 | `1.01214 @ 10000` | `step_10000` | `0.99697` | `0.09154` |
| `slot5` | `scratch` | `familiar` | 456 | 86,465,952 | `1.01285 @ 10000` | `step_10000` | `0.99917` | `0.09217` |
| `slot6` | `qwen` | `all_single` (13) | 42 | 87,428,102 | `0.95031 @ 8500` | `step_09000` | `0.95213` | `0.10294` |
| `slot7` | `scratch` | `all_single` (13) | 42 | 87,428,102 | `0.96845 @ 10000` | `step_10000` | `0.96688` | `0.11108` |

## 3. Frozen Test Results (Cluster Bootstrap `N=1000`, `K ∈ {0, 1, 2, 4}`, 5 Draws)

> **Small-seed caveat:** Familiar aggregates combine `S=3` model seeds (`42, 123, 456`); across-seed SE (`seed_se`) is noisy for `S=3` and must be read alongside family-cluster bootstrap 95% CIs (`[ci_low, ci_high]`) and R=20 Monte Carlo label-noise baselines.

### 3.1 Multi-Seed Familiar Models vs. Strongest Non-Neural Baseline (`query_material`) & All-Rules (`S=1`)
*(Test macro-mean NLL ↓, argmax accuracy ↑, vector Brier `Bvec` ↓, categorical Brier `Bcat` ↓, 10-bin ECE ↓, equal-material NLL `EqNLL` ↓)*

| Model / Baseline | Regime | K=0 NLL | K=1 NLL | K=2 NLL | K=4 NLL [95% CI] | Seed SE | Bvec (K4) | Bcat (K4) | Acc (K4) | ECE (K4) | EqNLL (K4) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `Qwen Familiar (S=3)` | `familiar` | `0.9619` | `0.9584` | `0.9532` | **`0.9533`** `[0.9285, 0.9768]` | `0.0035` | `0.0544` | `0.4679` | `0.678` | `0.0074` | **`0.9839`** |
| `Scratch Familiar (S=3)` | `familiar` | `1.0125` | `1.0121` | `1.0095` | `1.0099` `[0.9864, 1.0308]` | `0.0021` | `0.0886` | `0.5568` | `0.498` | `0.0088` | `0.9908` |
| `query_material` baseline | `familiar` | `0.9823` | — | — | `0.9823` | — | `0.0692` | `0.5146` | `0.680` | `0.0534` | `1.0013` |
| `Qwen All-Rules (slot6, S=1)` | `heldout`* | `0.9660` | `0.9140` | `0.9057` | **`0.9000`** | — | `0.1009` | `0.4835` | `0.578` | `0.0114` | — |
| `Scratch All-Rules (slot7, S=1)` | `heldout`* | `0.9775` | `0.9266` | `0.9161` | `0.9112` | — | `0.1068` | `0.4957` | `0.577` | `0.0065` | — |
| `Qwen Familiar (S=3)` | `heldout` | `1.1067` | `1.1077` | `1.1000` | `1.0990` `[1.0661, 1.1311]` | `0.0094` | `0.2018` | `0.5915` | `0.564` | `0.0897` | `1.0980` |
| `Scratch Familiar (S=3)` | `heldout` | `1.1958` | `1.1699` | `1.1595` | `1.1344` `[1.1013, 1.1659]` | `0.0225` | `0.2203` | `0.6400` | `0.476` | `0.0769` | `1.1522` |
| `query_material` baseline | `heldout` | `1.0218` | — | — | `1.0218` | — | `0.1652` | `0.5638` | `0.586` | `0.0314` | `1.0242` |
| `Qwen Familiar (S=3)` | `standard` | **`0.9203`** | `0.9285` | `0.9283` | `0.9259` `[0.8705, 0.9760]` | `0.0016` | `0.0491` | `0.3874` | `0.773` | `0.0214` | `0.9075` |
| `Scratch Familiar (S=3)` | `standard` | `0.9681` | `0.9690` | `0.9687` | `0.9688` `[0.9132, 1.0112]` | `0.0017` | `0.0766` | `0.4827` | `0.594` | `0.0135` | `0.9180` |
| `query_material` baseline | `standard` | `0.9426` | — | — | `0.9426` | — | `0.0577` | `0.4354` | `0.790` | `0.0736` | `0.9257` |

*\*Note:* Under `all_single` training (`slot6`, `slot7`), the 9 variants inside `heldout_transfer` were exposed in training, so they measure multi-task in-context rule identification rather than zero-shot unseen rules. Under `all_single`, `familiar_adaptation` test NLL at `K=4` is `1.0039` (`slot6`) and `1.0145` (`slot7`).

### 3.2 Remaining Non-Neural Baselines (`baselines_test.json`, K=4 clean NLL)
- `uniform`: `1.0986` across all regimes (`Bcat = 0.6667`).
- `training_prior`: `1.0953` (`familiar`), `1.0992` (`heldout`), `1.0996` (`standard`).
- `length_histogram`: `1.0932` (`familiar`), `1.0560` (`heldout`), `1.1144` (`standard`).
- `demo_majority`: `1.1597` (`familiar`), `1.1173` (`heldout`), `1.1863` (`standard`).
- `board_knn_final`: `1.1609` (`familiar`), `1.1084` (`heldout`), `1.1872` (`standard`).
- `board_knn_trajectories`: `1.6719` (`familiar`), `1.4317` (`heldout`), `1.6873` (`standard`).

### 3.3 Causal Intervention Controls (`K=4` NLL)
- **`Qwen Familiar (S=3)` (`familiar`):** clean `0.9533` vs. `token_matched_foreign` `0.9707` (`+0.0174` NLL degradation, worse than `K=0 = 0.9619`), `corrupted_trajectories` `0.9565`, `corrupted_outcomes` `0.9530`.
- **`Qwen All-Rules (slot6)` (`heldout` split):** clean `0.9000` vs. `token_matched_foreign` `0.9989` (`+0.0989`), `corrupted_trajectories` `0.9557` (`+0.0557`), `corrupted_outcomes` `0.8998`.
- **`Scratch All-Rules (slot7)` (`heldout` split):** clean `0.9112` vs. `token_matched_foreign` `1.0051` (`+0.0939`), `corrupted_trajectories` `0.9146`, `corrupted_outcomes` `0.9112`.
- **Mechanistic interpretation:** Under multi-rule exposure (`slot6`, `slot7`), in-context trajectory mechanics strongly identify the active variant rule (`antichess` `K=0->4`: `1.0173 -> 0.7646` Qwen / `1.0477 -> 0.7603` Scratch vs. `query_material` `1.4996`; `mobile_king`: `0.8258 -> 0.6224` Qwen / `0.9375 -> 0.6662` Scratch), so corrupting or mismatching trajectories degrades predictions back toward `K=0` or worse, while permuting demo outcomes (`corrupted_outcomes`) leaves rule mechanics intact.

### 3.4 `R=256` Re-Estimation Subset (`448` test roots, `0` censoring)
Across the `R=256` subset, vector Brier and held-out/standard NLLs decrease consistently with reduced Monte Carlo label-sampling variance:
- `slot0` (`qwen familiar`): `R=256` `K=4` NLL `1.0094` (`familiar`), `1.0885` (`heldout`), `0.8951` (`standard`), with vector Brier `0.0235 / 0.1973 / 0.0409` (vs. `R=20` full-test `0.0504 / 0.2084 / 0.0475`; debiased entropy floor `0.9751 / 0.7537 / 0.8328`).
- `slot6` (`qwen all_single`): `R=256` `K=4` NLL `1.0473` (`familiar`), `0.8817` (`heldout`), `0.9425` (`standard`).
- `slot7` (`scratch all_single`): `R=256` `K=4` NLL `1.0601` (`familiar`), `0.8974` (`heldout`), `0.9538` (`standard`).

## 4. Artifact & Code Verification
- Unit tests (`tests_v2/test_data_long_study.py`, `tests_v2/test_long_study_models.py`, `tests_v2/test_long_study_training.py`, `tests_v2/test_long_study_eval.py`): **29 passed, 0 failed, 0 skipped** (`325.02s`).
- Compact non-binary evaluation artifacts committed in `runs_v2/long_study/eval/`: `val_select_slot*.json`, `eval_summary_slot0..7.json`, `eval_agg_qwen_familiar_3seeds.json`, `eval_agg_scratch_familiar_3seeds.json`, `baselines_test.json`. (130 MB raw `eval_test_slot*.json` per-root files and weights/datasets are git-ignored.)


## 5. `K=8` & `K=16` Context-Length Scaling Extension
All `K=8` and `K=16` test prompts fit cleanly under `max_seq_len=32768` (`0` dropped demos across all `1,773` test roots; global max tokens `8,634` at `K=8`, `15,352` at `K=16`, `100.0%` root coverage).

- **New `K ∈ {0, 1, 2, 4, 8}` trained slots (`10,000` steps on `snap_v3`, `138,462,971` tokens each):**
  - `slot8` (`Qwen2.5-7B LoRA all_single`, `s42`): `best_val_nll = 0.95321 @ 10000` (`19,058 MB` peak VRAM); frozen full-val selection picked `step_10000` (`macro_nll = 0.95327`, `brier_vec = 0.10379`).
  - `slot9` (`Scratch 494M all_single`, `s42`): `best_val_nll = 0.96509 @ 10000` (`9,466 MB` peak VRAM); frozen full-val selection picked `step_09000` (`macro_nll = 0.96312`, `brier_vec = 0.10928`).

| Slot | Model / Train `K` | Regime | K=0 NLL | K=4 NLL | K=8 NLL | K=16 NLL | Foreign K=16 | R=256 K=8 | R=256 K=16 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `slot0` | Qwen Fam (`K≤4`) | `familiar` | `0.9573` | **`0.9480`** | `0.9509` | `0.9539` | `0.9701` | — | — |
| `slot0` | Qwen Fam (`K≤4`) | `standard` | **`0.9189`** | `0.9226` | `0.9262` | `0.9273` | `0.9273` | — | — |
| `slot6` | Qwen All (`K≤4`) | `heldout`* | `0.9660` | `0.9000` | `0.8998` | `0.8998` | `1.0036` | `0.8812` | `0.8811` |
| `slot7` | Scratch All (`K≤4`) | `heldout`* | `0.9775` | `0.9112` | `0.9089` | `0.9113` | `0.9707` | `0.8931` | `0.8923` |
| **`slot8`** | **Qwen All (`K≤8`)** | `heldout`* | `0.9627` | `0.8982` | **`0.8944`** | `0.8968` | `1.0012` | **`0.8760`** | `0.8785` |
| **`slot9`** | **Scratch All (`K≤8`)** | `heldout`* | `0.9665` | `0.9092` | **`0.9029`** | `0.9063` | `0.9857` | **`0.8867`** | `0.8877` |

- **Takeaway:** Zero-shot length extrapolation beyond `K=4` (`slot0`, `slot6`, `slot7`) saturates or ticks up slightly in macro NLL, whereas training with `K ∈ {0, 1, 2, 4, 8}` (`slot8`, `slot9`) unlocks additional `K=4 → K=8` gains (`0.8982 → 0.8944` Qwen, `0.9092 → 0.9029` Scratch, eliminating `slot6`'s zero-shot `K=8` regression on `dark_square_immunity`). Both `K=8`-trained models peak at their maximum trained context (`K=8`) and tick up slightly under unseen `K=16` extrapolation (`0.8968` / `0.9063`).
