import numpy as np
import pytest

from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs


# cas de reference de longstaff-schwartz
LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20)


def test_parite_call_put():
    c = prix_bs("call", 100, 95, 0.7, 0.03, 0.25, 0.02)
    p = prix_bs("put", 100, 95, 0.7, 0.03, 0.25, 0.02)
    assert c - p == pytest.approx(100 * np.exp(-0.014) - 95 * np.exp(-0.021), abs=1e-12)


def test_crr_europeen_egal_bs():
    assert prix_crr("put", exercice="europeen", **LS) == pytest.approx(prix_bs("put", **LS), abs=1e-3)


def test_crr_bermudien():
    eu = prix_bs("put", **LS)
    berm = prix_crr("put", exercice="bermudien", n_dates_exercice=50, **LS)
    am = prix_crr("put", **LS)
    assert eu < berm < am
    assert berm == pytest.approx(4.478, abs=1e-3)
