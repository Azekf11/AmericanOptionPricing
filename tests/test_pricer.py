import numpy as np
import pytest

from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs
from src.pricer.longstaff_schwartz import base, base_laguerre, prix_option_americaine, recurrence


# cas de reference de longstaff-schwartz
LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20)

# exemple a 8 trajectoires (poly, TD3)
HUIT = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34],
])

# put proche de totalenergies (q > r)
TTE = dict(S0=78.2, K=86.0, T=1.0, r=0.0244, sigma=0.234, q=0.0445)


# exemple du cours

def test_huit_trajectoires():
    flux, tau, _ = recurrence(HUIT, K=1.10, r=0.06, dt=1.0, type_option="put", degre=2)
    assert flux.mean() == pytest.approx(0.1144, abs=5e-5)
    assert list(tau) == [3, 3, 3, 1, 3, 1, 1, 1]


def test_huit_trajectoires_europeen():
    eu = np.maximum(1.10 - HUIT[:, -1], 0).mean() * np.exp(-0.18)
    assert eu == pytest.approx(0.0564, abs=5e-5)


# pricer

@pytest.mark.parametrize("itm", [True, False])
def test_lsm_cas_ls(itm):
    res = prix_option_americaine("put", itm_seulement=itm, **LS)
    berm = prix_crr("put", exercice="bermudien", n_dates_exercice=50, **LS)
    assert abs(res["prix"] - berm) < 3 * res["erreur_type"]
    assert res["prix"] > prix_bs("put", **LS)


def test_call_sans_dividende():
    res = prix_option_americaine("call", S0=40, K=40, T=1, r=0.06, sigma=0.2)
    assert abs(res["prix"] - prix_bs("call", 40, 40, 1, 0.06, 0.2)) < 3 * res["erreur_type"]


def test_call_avec_dividende():
    res = prix_option_americaine("call", S0=40, K=36, T=1, r=0.03, sigma=0.2, q=0.08)
    assert res["prix"] - prix_bs("call", 40, 36, 1, 0.03, 0.2, 0.08) > 3 * res["erreur_type"]


def test_ic_et_reproductibilite():
    a = prix_option_americaine("put", n_simulations=20_000, graine=7, **LS)
    b = prix_option_americaine("put", n_simulations=20_000, graine=7, **LS)
    assert a["prix"] == b["prix"]
    assert a["ic95"][0] < a["prix"] < a["ic95"][1]
    assert a["ic95"][1] - a["prix"] == pytest.approx(1.959964 * a["erreur_type"], rel=1e-6)


def test_exercice_immediat():
    res = prix_option_americaine("put", S0=10, K=40, T=1, r=0.06, sigma=0.2, n_simulations=10_000)
    assert res["prix"] == pytest.approx(30.0)


@pytest.mark.parametrize("modif", [
    dict(type_option="digital"), dict(sigma=-0.2), dict(degre=0),
    dict(methode_normale="numpy"), dict(generateur_uniforme="mersenne"),
])
def test_entrees_invalides(modif):
    p = dict(type_option="put", n_simulations=1000, **LS)
    p.update(modif)
    with pytest.raises(ValueError):
        prix_option_americaine(**p)


# variantes : plancher europeen, variable de controle, base de laguerre

def test_plancher_sans_effet_cas_ls():
    a = prix_option_americaine("put", **LS)
    b = prix_option_americaine("put", plancher_europeen=True, **LS)
    assert a["prix"] == b["prix"]


def test_plancher_corrige_prime_negative():
    brut = prix_option_americaine("put", **TTE)
    corr = prix_option_americaine("put", plancher_europeen=True, **TTE)
    assert brut["details"]["prime"] < -3 * brut["details"]["erreur_prime"]
    ref = prix_crr("put", exercice="bermudien", n_dates_exercice=50, **TTE) - prix_bs("put", **TTE)
    assert abs(corr["details"]["prime"] - ref) < 0.01
    assert corr["details"]["part_exercee"] < brut["details"]["part_exercee"] / 5


def test_controle_reduit_erreur():
    p = dict(S0=40, K=40, T=1, r=0.06, sigma=0.2, q=0.04)
    a = prix_option_americaine("call", plancher_europeen=True, **p)
    b = prix_option_americaine("call", plancher_europeen=True, controle_europeen=True, **p)
    assert b["erreur_type"] < a["erreur_type"] / 5


def test_call_sans_dividende_redevient_europeen():
    p = dict(S0=40, K=40, T=1, r=0.06, sigma=0.2, q=0.0)
    res = prix_option_americaine("call", plancher_europeen=True, controle_europeen=True, **p)
    assert res["prix"] == pytest.approx(prix_bs("call", **p), abs=1e-10)
    assert res["details"]["part_exercee"] == 0.0


def test_laguerre_formules():
    x = np.array([0.5, 1.0, 1.3])
    e = np.exp(-x / 2)
    attendu = np.column_stack([np.ones_like(x), e, e * (1 - x), e * (1 - 2 * x + x**2 / 2)])
    assert np.allclose(base_laguerre(x, 3), attendu)
    assert base_laguerre(x, 3).shape == base(x, 3).shape


def test_laguerre_prix():
    berm = prix_crr("put", **LS, exercice="bermudien", n_dates_exercice=50)
    res = prix_option_americaine("put", **LS, base_regression="laguerre")
    assert abs(res["prix"] - berm) < 3 * res["erreur_type"]
