from math import erf, sqrt

import numpy as np
import pytest

from src.modele.arbre_crr import prix_crr
from src.pricer.bornes import N_rapide, borne_haute_duale


LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20)


def test_N_rapide():
    x = np.linspace(-6, 6, 2001)
    exact = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x])
    assert np.max(np.abs(N_rapide(x) - exact)) < 1e-7


# la borne haute doit etre au-dessus du prix exact, et proche
@pytest.mark.parametrize("type_option, p", [
    ("put", LS),
    ("call", dict(S0=40.0, K=40.0, T=1.0, r=0.06, sigma=0.2, q=0.08)),
])
def test_borne_haute(type_option, p):
    berm = prix_crr(type_option, **p, exercice="bermudien", n_dates_exercice=50)
    res = borne_haute_duale(type_option, **p, n_regression=50_000, n_exterieur=5_000, noeuds=20)
    assert res["prix"] > berm - 3 * res["erreur_type"]
    assert res["prix"] < berm + 0.01
