import pandas as pd
import numpy as np

# --- Business assumptions (keep in sync with docs/problem_statement.md) ---
OFFER_COST = 50        # cost of one retention offer ($)
CUSTOMER_VALUE = 780   # revenue kept if a churner is saved ($): about $65 x 12 months
SUCCESS_RATE = 0.30    # share of contacted churners the offer actually saves

GAIN_TP = round(SUCCESS_RATE * CUSTOMER_VALUE - OFFER_COST, 2)   # +184 per churner reached
COST_FP = OFFER_COST

def net_savings(y_true, y_pred):
    """Money gained versus doing nothing."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return GAIN_TP * tp - COST_FP * fp


def savings_curve(y_true, proba, thresholds=None):
    """Net savings for each possible cut-off."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [(t, net_savings(y_true, (proba >= t).astype(int))) for t in thresholds]
    return pd.DataFrame(rows, columns=["threshold", "net_savings"])