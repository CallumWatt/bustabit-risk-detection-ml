# Bustabit Risky-Gambling Behaviour Detection

Machine learning pipeline that flags risky gambling behaviour from bet-level activity on **Bustabit** (a crash-style betting game). The project turns raw, per-bet records into player-level behavioural features, assigns risk labels, and trains and compares several classifiers to identify high-risk players — with an emphasis on **recall**, so risky players aren't missed. It also includes a forward-looking workflow that predicts which players are likely to *escalate* their betting in future.

## Motivation

Detecting problematic gambling early is a core responsible-gambling problem: the goal isn't to predict who wins, but to surface players whose behaviour (rising stakes, high loss rates, heavy volume) suggests elevated risk, so they can be supported before harm compounds.

## Dataset

`bustabit.csv` — ~50,000 individual bets with the following fields:

| Field | Description |
|-------|-------------|
| `GameID` / `Username` | Game and player identifiers |
| `Bet` | Amount staked |
| `CashedOut` | Multiplier the player cashed out at (if they did) |
| `Bonus` / `Profit` | Bonus applied and net profit/loss on the bet |
| `BustedAt` | Multiplier the game crashed at |
| `PlayDate` | Timestamp of the bet |

Missing values are handled during loading (e.g. an un-cashed bet is treated as a full loss of the stake), and bets are ordered per player to preserve betting sequence.

## Feature engineering

Bet-level rows are aggregated into one row per player, capturing both scale and behaviour:

- **Staking:** average, median, max and variance of bet size
- **Outcomes:** total and average profit, loss rate (share of losing bets)
- **Intensity:** number of bets placed
- **Risk ratio:** `max_bet / avg_bet`, capturing outsized single stakes
- **Escalation trend:** log-linear slope of stake size over bet order (via linear regression on `log(bet)`), measuring whether a player ramps their stakes up over time

## Two workflows

**1. Risk classification (multiclass).**
A transparent, rule-based scoring scheme combines quantile thresholds (75th/90th/98th percentiles) across loss rate, bet-risk ratio, betting volume and escalation trend — with extra weight for players in the worst 5% by total loss — to label each player **low / medium / high risk**. Classifiers are then trained to reproduce and generalise these labels.

**2. Future escalation prediction (binary).**
Each player's history is split 70/30 into past and future. Features are computed only from the *past*, while the label is whether the player's stake-escalation trend in the *future* falls in the top 10%. This tests whether early behaviour can predict later escalation, rather than just describing behaviour after the fact.

## Modelling & evaluation

- **Models compared:** Logistic Regression, Random Forest, SVC, K-Nearest Neighbours
- **Setup:** stratified train/test split, `StandardScaler`, `class_weight="balanced"` to handle class imbalance, and 5-fold cross-validation
- **Metrics:** accuracy, precision, recall, F1 (weighted and macro), cross-validated F1, PR-AUC and precision–recall curves, with recall for the high-risk class prioritised
- **Error analysis:** false positives and false negatives for the target class are inspected, including their mean feature profiles

## Interpretability

- **SHAP** explains *why* players are flagged — `TreeExplainer` for the Random Forest and `KernelExplainer` for the SVC — highlighting the strongest drivers of high-risk predictions
- **Feature importances** from the Random Forest complement the SHAP view

## Unsupervised analysis

**KMeans** clustering (with the elbow method and a 2D **PCA** projection) segments players into behavioural groups, which are cross-tabulated against both the rule-based risk labels and the model predictions to check how well natural clusters line up with the risk framework.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · SHAP · Matplotlib · seaborn

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn
python bustabitML.py
```

Place `bustabit.csv` in the same directory. The script runs both workflows end to end, printing model-comparison tables and displaying the plots (risk distributions, correlation matrix, PR curves, SHAP summaries, clustering).

## Note

This is a personal learning project built on a public dataset for educational purposes. The risk labels are heuristic and illustrative, not a validated clinical or commercial risk model.
