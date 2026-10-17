# Model Card: Telco Churn Decision Model

A model card is a short, honest description of what a model is for, how it was built, how well it works, and where it should not be trusted.

## Model details

- **Type:** Logistic regression inside a scikit-learn `Pipeline` (standard scaling for 4 numeric columns, one-hot encoding for 15 categorical columns, 45 input features after encoding).
- **Task:** Binary classification. Output is the probability that a customer will churn, plus a recommendation to contact or not contact them.
- **Decision rule:** Contact a customer when the predicted churn probability is at least **0.25**. This cut-off was chosen to maximise net savings under the cost assumptions below, using out-of-fold predictions on the training data only.
- **Code:** `src/train.py` rebuilds the model and the cut-off from the raw data.

## Intended use

- Ranking and triaging telecom customers for a retention campaign.
- Demonstrating cost-based decision making in a machine learning portfolio project.

## Out-of-scope use

- Any decision with legal or major financial consequences for an individual (credit, pricing, denying service).
- Use on a different company, country or market without retraining and re-validating. The training data is a public sample dataset, not data from a Kenyan operator.
- Treating SHAP explanations as proof of what *causes* churn.

## Data

- IBM Telco Customer Churn sample dataset: 7,043 customers, 21 columns, 26.5% churned.
- Split 80/20, stratified on churn (5,634 training, 1,409 test), random seed 42.
- Preprocessing issue handled: `TotalCharges` was stored as text and blank for 11 customers with tenure 0; blanks were set to 0.
- `customerID` is excluded from the features.

## Cost assumptions behind the decision rule

| Assumption | Value |
|---|---|
| Retention offer cost | $50 |
| Value of a saved customer | $780 (about $65 x 12 months) |
| Offer success rate | 30% |

These are **assumptions**, not measured values. Changing them changes the best cut-off.

## Performance

Cross-validated on training data (5 folds): accuracy 0.802, precision 0.653, recall 0.543, F1 0.592, ROC-AUC 0.846 (at the default 0.50 cut-off).

On the held-out test set at the 0.25 cut-off:

| | Value |
|---|---|
| Customers | 1,409 (374 churners) |
| Contacted | 609 |
| Churners reached | 303 (81%) |
| Wasted offers | 306 |
| Net savings (assumed costs) | $40,452 |

Comparison on the same test set: contact nobody $0, contact everybody $17,066, model at the 0.50 cut-off $33,006.

Other models tried (random forest, XGBoost, class-weighted logistic regression) reached the same ROC-AUC (0.845 to 0.847), so the simplest and most explainable model was kept.

## Explainability

SHAP values from a linear explainer. Strongest drivers: tenure (short tenure raises risk), contract type (month-to-month raises risk, two-year lowers it), and fiber optic internet (raises risk). Charges columns share credit with tenure and service type, so their individual effects should not be over-read. The summary chart is in `docs/shap_summary.png`.

## Ethical considerations and fairness

- **Sensitive inputs:** `gender` and `SeniorCitizen` are used by the model. **No fairness audit has been done.** Before any real use, performance (recall, precision, offer rates) should be compared across gender and age groups, and removing these inputs should be tested.
- **Who is affected:** the model decides who receives a retention offer, which may mean some customers get discounts others do not. This should be reviewed for fairness and transparency.
- **Privacy:** a real deployment would handle customer data under applicable data-protection law (for example Kenya's Data Protection Act, 2019). This project uses only a public sample dataset.

## Limitations

- Costs and success rate are assumed. A real A/B test would be needed to measure them.
- Single dataset, single train/test split, no confidence intervals.
- Probabilities have not been checked for calibration, and the cost maths relies on them.
- A snapshot dataset has no time dimension, so drift over time cannot be assessed.
- The demo app estimates total charges (months x monthly charges) and does not validate that the entered charges match the chosen services.

## Maintenance (if this were deployed)

- Monitor the share of customers flagged, the input distributions and the observed churn of contacted versus non-contacted groups.
- Retrain on a schedule, and re-select the cut-off whenever costs or offer success rates change.