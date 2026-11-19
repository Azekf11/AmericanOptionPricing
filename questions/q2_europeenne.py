# question 2 : americaine contre europeenne equivalente
# a. put, grille de l'article de longstaff-schwartz
# b. call, effet du dividende q
# c. totalenergies (call et put, 3 strikes) + sensibilite au taux
# lancer depuis la racine :  python -m questions.q2_europeenne

from math import exp, log, pi, sqrt

import numpy as np

from questions.outils import CAS_LS, crr_americain, crr_bermudien, ecrire_csv
from src.calibration.totalenergies import EURIBOR_12M, calibrer
from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import N, prix_bs
from src.pricer.longstaff_schwartz import prix_option_americaine


N_SIM = 100_000


# une ligne de comparaison
# brut   : algorithme du sujet tel quel
# corrige: plancher europeen + variable de controle europeenne
def comparer(type_option, **p):
    eu = prix_bs(type_option, **p)
    berm = crr_bermudien(type_option, **p)
    brut = prix_option_americaine(type_option, **p, n_simulations=N_SIM)
    corr = prix_option_americaine(type_option, **p, n_simulations=N_SIM,
                                  plancher_europeen=True, controle_europeen=True)

    # aucun exercice anticipe -> prix europeen exact, erreur nulle
    if corr["erreur_type"] > 1e-10:
        ecart_corr = (corr["prix"] - berm) / corr["erreur_type"]
    else:
        ecart_corr = 0.0

    return {
        "europeen_bs": eu,
        "crr_bermudien": berm,
        "crr_americain": crr_americain(type_option, **p),
        "prime_crr": berm - eu,
        "lsm_brut": brut["prix"],
        "se_brut": brut["erreur_type"],
        "prime_brut": brut["details"]["prime"],
        "se_prime_brut": brut["details"]["erreur_prime"],
        "exerce_brut": brut["details"]["part_exercee"],
        "ecart_brut_en_se": (brut["prix"] - berm) / brut["erreur_type"],
        "lsm_corrige": corr["prix"],
        "se_corrige": corr["erreur_type"],
        "prime_corrige": corr["details"]["prime"],
        "exerce_corrige": corr["details"]["part_exercee"],
        "ecart_corrige_en_se": ecart_corr,
    }


def grille_put():
    lignes = []
    for S0 in (36.0, 40.0, 44.0):
        for sigma in (0.2, 0.4):
            for T in (1.0, 2.0):
                p = dict(S0=S0, K=40.0, T=T, r=0.06, sigma=sigma, q=0.0)
                lignes.append({"S0": S0, "sigma": sigma, "T": T, **comparer("put", **p)})
                print("put", S0, sigma, T, round(lignes[-1]["prime_brut"], 4))
    ecrire_csv("q2_put_grille", lignes)


def call_dividende():
    lignes = []
    for q in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        p = dict(S0=40.0, K=40.0, T=1.0, r=0.06, sigma=0.2, q=q)
        lignes.append({"q": q, **comparer("call", **p)})
        print("call q =", q, round(lignes[-1]["prime_crr"], 4))
    ecrire_csv("q2_call_dividende", lignes)


def totalenergies():
    cal = calibrer()
    ecrire_csv("calibration", [{
        "date": cal["date"], "S0": cal["S0"], "sigma": cal["sigma"],
        "sigma_brut": cal["sigma_brut"], "q": cal["q"], "r": cal["r"],
        "dividendes_12m": cal["dividendes_12m"],
        "detail": " ".join(f"{d}:{m}" for d, m in cal["detail_dividendes"]),
    }])

    lignes = []
    for type_option in ("call", "put"):
        for K in (70.0, 78.0, 86.0):
            p = dict(S0=cal["S0"], K=K, T=1.0, r=cal["r"], sigma=cal["sigma"], q=cal["q"])
            lignes.append({"option": type_option, "taux": "estr", "K": K, **comparer(type_option, **p)})
            print("TTE", type_option, K, round(lignes[-1]["prime_brut"], 4))

    # meme chose a la monnaie avec l'euribor 12 mois
    for type_option in ("call", "put"):
        p = dict(S0=cal["S0"], K=78.0, T=1.0, r=EURIBOR_12M, sigma=cal["sigma"], q=cal["q"])
        lignes.append({"option": type_option, "taux": "euribor", "K": 78.0, **comparer(type_option, **p)})

    ecrire_csv("q2_totalenergies", lignes)


# delta et gamma du put LS : differences finies centrees, memes tirages de part et d'autre
def grecques(h=0.5):
    deltas, gammas = [], []
    for g in range(1, 11):
        v = {}
        for cle, choc in (("bas", -h), ("milieu", 0.0), ("haut", h)):
            p = dict(CAS_LS, S0=CAS_LS["S0"] + choc)
            v[cle] = prix_option_americaine("put", **p, n_simulations=200_000, graine=g, antithetique=True)["prix"]
        deltas.append((v["haut"] - v["bas"]) / (2 * h))
        gammas.append((v["haut"] - 2 * v["milieu"] + v["bas"]) / h**2)

    arbre = {choc: prix_crr("put", **dict(CAS_LS, S0=CAS_LS["S0"] + choc), n_pas=4000) for choc in (-h, 0.0, h)}

    # europeen : formules fermees
    S0, K, T, r, sigma = CAS_LS["S0"], CAS_LS["K"], CAS_LS["T"], CAS_LS["r"], CAS_LS["sigma"]
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

    ligne = {
        "delta_lsm": float(np.mean(deltas)), "delta_lsm_se": float(np.std(deltas, ddof=1) / sqrt(len(deltas))),
        "gamma_lsm": float(np.mean(gammas)), "gamma_lsm_se": float(np.std(gammas, ddof=1) / sqrt(len(gammas))),
        "delta_arbre": (arbre[h] - arbre[-h]) / (2 * h),
        "gamma_arbre": (arbre[h] - 2 * arbre[0.0] + arbre[-h]) / h**2,
        "delta_europeen": N(d1) - 1,
        "gamma_europeen": exp(-0.5 * d1**2) / sqrt(2 * pi) / (S0 * sigma * sqrt(T)),
    }
    print({k: round(v, 4) for k, v in ligne.items()})
    ecrire_csv("q2_grecques", [ligne])


def main():
    print("Q2")
    grille_put()
    call_dividende()
    totalenergies()
    grecques()


if __name__ == "__main__":
    main()
