import numpy as np

from src.evaluation import (COST_FP, CUSTOMER_VALUE, GAIN_TP, OFFER_COST,
                            SUCCESS_RATE, net_savings, savings_curve)


def test_gain_per_caught_churner_follows_the_assumptions():
    assert GAIN_TP == round(SUCCESS_RATE * CUSTOMER_VALUE - OFFER_COST, 2)
    assert COST_FP == OFFER_COST


def test_net_savings_on_a_known_example():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]          # 1 churner reached, 1 wasted offer
    assert net_savings(y_true, y_pred) == GAIN_TP - COST_FP


def test_contacting_nobody_saves_nothing():
    assert net_savings([1, 0, 1, 0], [0, 0, 0, 0]) == 0


def test_perfect_targeting_reaches_every_churner():
    y = np.array([1, 1, 1, 0, 0])
    assert net_savings(y, y) == 3 * GAIN_TP


def test_savings_curve_finds_a_sensible_cutoff():
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1, 0.3])
    curve = savings_curve(y, proba)
    assert list(curve.columns) == ["threshold", "net_savings"]
    assert curve["net_savings"].max() == 2 * GAIN_TP
