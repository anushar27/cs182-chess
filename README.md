# In-Context Learning of Chess Variants

CS 182 final project. Can a model pretrained on standard chess correctly
evaluate a familiar board position after a small rule change, given the new
rules and K example games in its context?

## Setup

Use Python 3.10 or later. Run these commands from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -c "from chess_icl import data, model, evaluate"
```

The scaffold uses the standard library. Add dependencies when the team chooses
its training framework, game parser, and reference evaluator. Use `data/` for
local datasets and `runs/` for checkpoints and results.

## Experiment

1. Pretrain a position evaluator on standard chess. Keep the trained weights
   frozen for the evaluation runs.
2. Choose one small rule change. Build a held-out set of positions valid under
   both rulesets and label each position independently as win, draw, or loss
   under each ruleset, from the side-to-move perspective. Use matching board
   state and side to move, and record any history required to determine the
   outcomes. Choose a reference evaluator that supports the variant.
3. Run the same positions through the same model in three conditions:
   - Standard rules, zero example games.
   - Variant rules, zero example games.
   - Variant rules and K complete variant games in context. Start with the
     proposal's K values of 1, 2, 4, 8, and 16 where context length permits.
4. Report variant accuracy for each K on the full set and on positions whose
   reference labels differ between rulesets. Include the sample count for the
   changed-label subset and identify any direct win-to-loss or loss-to-win
   reversals. Compare K-shot accuracy with the variant rules-only baseline.

Keep the evaluation set fixed across runs. Draw example games from a separate
pool, keeping source games and evaluation positions out of the demonstrations.
Record the demo selection seed and use nested sets across K within each seed
so the K-shot run adds games to the smaller context. Record the reference
labeling method, including any thresholds or search budget.

## Implementation handoff

| File | Team implementation |
| --- | --- |
| [data.py](chess_icl/data.py) | Load games, prepare position labels, and build the paired evaluation set. |
| [model.py](chess_icl/model.py) | Choose the model, train on standard chess, and define how to encode rules, example games, and the query position. |
| [evaluate.py](chess_icl/evaluate.py) | Sample games, run each condition, and report outcome accuracy. |

The game record stores the result of the completed demonstration game. Paired
evaluation labels describe the position under each ruleset. Decide how
standard-chess training positions get their targets as part of pretraining.
The proposal's small transformer is a starting point. The revised experiment
uses explicit rules, standard-chess pretraining, and a single held-out variant.
