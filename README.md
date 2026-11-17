# In-Context Learning of Chess Variants

CS 182 final project. Can a model pretrained on standard chess correctly
evaluate a familiar board position after a small rule change, given the new
rules and K example games in its context?

## Setup

Use Python 3.10 or later. Run these commands from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -c "from chess_icl import data, model, evaluate, prepare_data"
```

The scaffold imports with the standard library. Use `data/` for local datasets
and `runs/` for checkpoints and results.

## Offline data preparation

1. Define the rule change in [configs/variants.ini](configs/variants.ini).
   The template is commented out until a rule is chosen. Use the same file
   and variant name in both tools. Check that the rule is supported by
   [Fairy-Stockfish's variant configuration](https://fairy-stockfish.github.io/custom-variants/).
2. Install the data-only Python dependency with
   `python -m pip install -r requirements-data.txt`. Install a
   [Fairy-Stockfish binary](https://github.com/fairy-stockfish/Fairy-Stockfish/releases)
   separately. Check the config with `/path/to/fairy-stockfish check configs/variants.ini`
   once the variant is defined.
3. Generate demonstration games with pyffish. Load the INI text, pick moves
   from its legal move list using a policy the team chooses, and use the
   variant's terminal result to create a game record. If you want
   engine-played games, Fairy-Stockfish can also supply the move policy.
4. Use Fairy-Stockfish over UCI to score each held-out position twice.
   Select `chess` for the standard score and the configured variant for
   the variant score. Check that both tools agree on legal moves and
   terminal outcomes for positions affected by the rule change.

Record both tool versions, the move policy, and the engine search settings
alongside the saved datasets. Pretraining and evaluation consume the saved
games and scores. At inference the model gets the natural-language rules,
the example games, and the query position.

## Experiment

1. Pretrain a position evaluator to predict standard-chess engine scores in
   centipawns. Keep the trained weights frozen for the evaluation runs.
2. Choose one small rule change. Build a held-out set of positions valid under
   both rulesets and obtain a reference score under each ruleset. Positive
   scores favor the side to move, negative scores favor the opponent, and
   scores near zero indicate an even evaluation. Store separate standard and
   variant FENs with matching piece placement and side to move. Preserve
   variant state such as Crazyhouse reserves and any replay history required
   by each ruleset. Document the reference evaluator's score scale. Define a consistent
   mapping for mates and other decisive terminal scores into finite values
   before training or measuring error.
3. Run the same positions through the same model in three conditions:
   - Standard rules, zero example games.
   - Variant rules, zero example games.
   - Variant rules and K complete variant games in context. Start with the
     proposal's K values of 1, 2, 4, 8, and 16 where context length permits.
4. Report mean absolute error in centipawns against the matching reference
   score for each condition. Also report variant error on positions where the
   reference scores favor opposite sides across rulesets. Define a near-zero
   margin so small fluctuations around equality do not count as reversals,
   and report the subset size. Compare each K with the variant rules-only
   baseline on the same positions.

Keep the evaluation set fixed across runs. Draw example games from a separate
pool, keeping source games and evaluation positions out of the demonstrations.
Record the demo selection seed and use nested sets across K within each seed
so the K-shot run adds games to the smaller context. Record the reference
scoring method, including the search budget and score conversion.

## Implementation handoff

| File | Team implementation |
| --- | --- |
| [prepare_data.py](chess_icl/prepare_data.py) | Generate games offline and obtain Fairy-Stockfish reference scores. |
| [data.py](chess_icl/data.py) | Load saved games and paired reference position scores. |
| [model.py](chess_icl/model.py) | Choose the model, train on standard chess, and define how to encode rules, example games, and the query position. |
| [evaluate.py](chess_icl/evaluate.py) | Sample games, run each condition, and report centipawn error. |

The game record stores the completed demonstration game's result as metadata.
The model predicts a numeric score for the queried position.
The proposal's small transformer is a starting point. The revised experiment
uses explicit rules, standard-chess pretraining, and a single held-out variant.
