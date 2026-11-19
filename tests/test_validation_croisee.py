from math import sqrt

from src.pricer.longstaff_schwartz import prix_option_americaine
from validation import pricer_lsm as autre


LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20)


def test_lcg_du_second_pricer():
    g = autre.GenerateurCongruentiel(2026)
    rapides = [int(v) for v in g.entiers(5_000)]
    x, naifs = 2026, []
    for _ in range(5_000):
        x = (g.A * x + g.C) % g.M
        naifs.append(x)
    assert rapides == naifs


# deux codes independants, meme option : prix compatibles
def test_deux_pricers_d_accord():
    a = prix_option_americaine("put", **LS)
    b = autre.prix_americaine_lsm(spot=36, strike=40, volatilite=0.2, taux=0.06, maturite=1.0,
                                  type_option="put", degre=3, graine=1)
    assert abs(a["prix"] - b["prix"]) < 4 * sqrt(a["erreur_type"] ** 2 + b["erreur_standard"] ** 2)


def test_arbres_d_accord():
    from src.modele.arbre_crr import prix_crr
    nous = prix_crr("put", **LS, n_pas=2000, exercice="bermudien", n_dates_exercice=50)
    eux = autre.prix_arbre_binomial(spot=36, strike=40, volatilite=0.2, taux=0.06, maturite=1.0,
                                    type_option="put", n_pas=2000, n_dates_exercice=50)
    assert abs(nous - eux) < 1e-3
