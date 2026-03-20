# Phase 2: Chess → Logistics Transfer Learning

## Research Question

Does a chess-trained AlphaZero trunk (spatial planning, multi-step lookahead,
positional evaluation) transfer useful representations to logistics offer ranking,
compared to training the same architecture from scratch on logistics data?

Secondary question: Does the concept bottleneck (interpretable intermediate layer)
improve logistics prediction accuracy, or only interpretability?

---

## Experiment Design

Train and compare five models on the same logistics dataset(s):

| Model | Description |
|-------|-------------|
| **A — XGBoost baseline** | Standard tabular baseline; establishes floor |
| **B — MLP scratch** | Fully connected net, matched parameter count to D; no spatial inductive bias |
| **C — Chess trunk frozen** | LogisticsInputAdapter → frozen chess res_tower → new heads |
| **D — Chess trunk fine-tuned** | Same as C but end-to-end fine-tuning |
| **E — Chess trunk + concept supervision** | D + logistics concept labels mapped from chess concepts |

If D or E beats B, transfer learning works. If E beats D, supervised concepts help.

---

## Target Tasks

### Task 1: On-Time Delivery Prediction (binary classification)
- Input: shipment features at offer/booking time
- Output: probability shipment arrives on time
- Metric: AUC-ROC, F1

### Task 2: Route Ranking (learning-to-rank)
- Input: a set of candidate routes for a given shipment
- Output: ranked list by expected margin × on-time probability
- Metric: NDCG@5, MRR

### Task 3: Delay Duration Regression (optional)
- Input: same shipment features
- Output: predicted hours late (if late)
- Metric: MAE, RMSE

---

## Public Datasets (Scratch Experiments)

### Primary

**Cargo 2000 (C2K)** — IATA air cargo tracking data
- Source: https://www.kaggle.com/datasets/crawford/cargo-2000-dataset
- ~3,600 real shipments, planned vs actual milestone timestamps
- Features: origin airport, destination airport, cargo type, planned transit
- Target: on-time delivery per milestone
- Best fit: Task 1 + Task 3

**DataCo Smart Supply Chain**
- Source: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
- ~180k orders, shipping mode, origin/destination region, product category, order value
- Target: late_delivery_risk (binary), days_for_shipping_real
- Best fit: Task 1 + Task 3 (large volume, good for generalization testing)

**Amazon Last-Mile Routing Challenge**
- Source: https://registry.opendata.aws/amazon-last-mile-challenges/
- 6,112 historical routes with actual stop sequences and scores
- Best fit: Task 2 (route ranking / sequencing)

### Secondary (augment if needed)

**SupplyGraph** — FMCG supply chain, graph-structured
- Source: https://github.com/ciol-researchlab/SupplyGraph
- Useful if we want to test GNN variants alongside the trunk approach

**time:matters internal data** (Phase 2b, pending access)
- Offer features: origin, destination, cargo_class, weight_kg, deadline_hours,
  declared_value, dgr_class, service_tier
- Targets: on_time (binary), margin (float), selected_route (ranking label)
- This is the real target domain; public datasets validate the method first

---

## Architecture

### LogisticsInputAdapter
Maps tabular offer features → spatial tensor that the chess trunk can process.

```
offer_features (N,)
    → FeatureEmbedding (linear + BN per feature group)
    → reshape to (256, 8, 8)   # 256 channels, spatial grid
    → Conv1x1 → AZ_CHANNELS × 8 × 8
```

The 8×8 grid is not geographically meaningful — it gives the res_tower its expected
input shape. Spatial convolutions will learn to route information across feature
groups, which is the inductive bias we want to test.

### Heads (new, trained from scratch in all experiments)

```python
class OnTimeHead(nn.Module):      # binary classification
    # global_avg_pool → Linear(AZ_CHANNELS, 1) → sigmoid

class RouteRankingHead(nn.Module): # top-5 route scores
    # global_avg_pool → Linear(AZ_CHANNELS, 5)

class LogisticsConcepts(nn.Module): # mirrors chess ConceptBottleneck
    # maps chess concepts → logistics analogues (see below)
```

### Concept Mapping (chess → logistics)

| Chess Concept       | Logistics Analogue         | Label Source                        |
|---------------------|----------------------------|-------------------------------------|
| material_balance    | margin_headroom            | (declared_value - cost) / declared_value |
| king_safety         | critical_node_risk         | 1 - hub_redundancy_score            |
| piece_mobility      | route_optionality          | n_available_routes / max_routes     |
| pawn_structure      | supply_chain_dependency    | single_carrier_concentration        |
| space_control       | network_coverage           | origin_dest_connectivity_score      |
| tactical_threat     | disruption_probability     | dgr_flag OR historical_delay_rate   |

---

## Repo Structure

```
chess-logistics/
├── data/
│   ├── cargo2000/          # raw + processed
│   ├── dataco/             # raw + processed
│   └── amazon_lastmile/    # raw + processed
├── src/
│   ├── adapter.py          # LogisticsInputAdapter
│   ├── heads.py            # OnTimeHead, RouteRankingHead, LogisticsConcepts
│   ├── concepts.py         # compute_logistics_concept_labels()
│   ├── dataset.py          # DataLoader for each dataset
│   ├── train.py            # training loop (models A–E)
│   └── evaluate.py         # AUC, NDCG@5, MRR, MAE
├── experiments/
│   ├── run_xgboost.py      # Model A
│   ├── run_mlp_scratch.py  # Model B
│   ├── run_transfer.py     # Models C, D, E (flag --mode frozen|finetune|concept)
│   └── results/            # JSON logs, plots
├── notebooks/
│   └── exploration.ipynb   # EDA on each dataset
├── chess_net.py            # symlink or copy from Chess-Sim
└── README.md
```

---

## Success Criteria

| Finding | Conclusion |
|---------|------------|
| D beats B by >2% AUC | Transfer learning works; spatial chess representations generalize |
| E beats D | Supervised concept bottleneck adds predictive value |
| C competitive with D | Trunk features useful without fine-tuning (strong transfer) |
| A beats B and D | Tabular inductive bias dominates; spatial approach not worth it |

Any outcome is a valid research result. The goal is to know, not to confirm.

---

## Steps to Start

1. Create new repo `chess-logistics` (separate from Chess-Sim)
2. Download Cargo 2000 and DataCo datasets (both free on Kaggle, ~10–50MB each)
3. EDA notebook: understand feature distributions, missing data, class balance
4. Implement `dataset.py` — standardize features, train/val/test split
5. Implement and run Model A (XGBoost) to establish baseline AUC
6. Implement `LogisticsInputAdapter` + `OnTimeHead`
7. Run Model B (MLP scratch) — match parameter count to chess trunk + heads
8. Copy `chess_net.py` from Chess-Sim; load `model.pt` weights
9. Run Model C (frozen) and D (fine-tuned)
10. Implement `concepts.py`, run Model E
11. Compare all five on held-out test set; write up results

---

## Notes

- Start with Task 1 (on-time binary) only — simplest target, clear metric
- Add Task 2 (ranking) only if Task 1 shows transfer signal worth pursuing
- time:matters internal data is Phase 2b — validate method on public data first
  so results are reproducible and publishable regardless of data access
- Keep chess-logistics repo standalone: copy (not symlink) chess_net.py so
  the logistics project has no runtime dependency on Chess-Sim
