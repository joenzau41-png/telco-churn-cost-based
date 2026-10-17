# Problem Statement

## Business problem
A telecom company loses revenue when customers cancel (churn). It can send a retention offer to customers at risk, but offers cost money and do not always work. The goal is to decide **which customers to contact** so that total savings are as high as possible.

## Stakeholders
- **Retention team:** acts on the list of customers to contact.
- **Finance:** owns the offer budget.
- **Data team:** builds and maintains the model.

## Success metric
Net savings on customers the model has never seen, compared with two simple strategies: contact nobody and contact everybody. Accuracy alone is not used: 73.5% of customers stay, so a model that always predicts "stays" is 73.5% accurate and catches no churners.

## Cost assumptions
| Assumption | Value |
|---|---|
| Retention offer cost | $50 |
| Value of a saved customer | $780 (about $65 x 12 months) |
| Offer success rate | 30% |

Net savings = $184 x (churners reached) - $50 x (wasted offers). The break-even churn probability is about 21%.

## Constraints and risks
- The costs above are assumptions, not measured values.
- The data is a public sample, not a live company dataset.
- Sensitive inputs (gender, senior citizen) need a fairness review before real use.