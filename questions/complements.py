# verifications en plus des trois questions
# lancer depuis la racine :  python -m questions.complements

from math import exp, sqrt

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.aleatoire.normales import Normales
from src.calibration.totalenergies import calibrer
from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs, prix_bs_vect
from src.modele.payoff import payoff
from src.modele.trajectoires import simuler_trajectoires
from src.pricer.longstaff_schwartz import frontiere, prix_option_americaine, recurrence


# plancher europeen sur 10 graines
def plancher(cal):
    cas = {
        "put LS": ("put", dict(CAS_LS)),
        "put TTE K=86": ("put", dict(S0=cal["S0"], K=86.0, T=1.0, r=cal["r"],
                                     sigma=cal["sigma"], q=cal["q"])),
        "call q=4%": ("call", dict(S0=40.0, K=40.0, T=1.0, r=0.06, sigma=0.2, q=0.04)),
    }
    lignes = []

    for nom, (type_option, p) in cas.items():
        prime_ref = crr_bermudien(type_option, **p) - prix_bs(type_option, **p)
        for mode, options in (("brut", {}), ("plancher", {"plancher_europeen": True})):
            primes, parts = [], []
            for g in range(1, 11):
                r = prix_option_americaine(type_option, **p, graine=g, **options)
                primes.append(r["details"]["prime"])
                parts.append(r["details"]["part_exercee"])
            m = float(np.mean(primes))
            s = float(np.std(primes, ddof=1) / sqrt(len(primes)))
            lignes.append({"cas": nom, "estimateur": mode, "prime_moyenne": m, "se_moyenne": s,
                           "prime_crr": prime_ref, "ecart_en_se": (m - prime_ref) / s,
                           "part_exercee": float(np.mean(parts))})
            print(nom, mode, round(m, 4), round((m - prime_ref) / s, 1))

    ecrire_csv("comp_plancher", lignes)


# diagnostic de l'exercice (put LS et put totalenergies)
def diagnostic(cal):
    lignes = []

    # put LS : prix critique de l'arbre bermudien et de la regle LSM
    p = CAS_LS
    _, front_crr = prix_crr("put", **p, n_pas=2000, exercice="bermudien",
                            n_dates_exercice=50, frontiere=True)
    res = prix_option_americaine("put", **p)
    front_lsm = frontiere(res["details"]["coefs"], p["K"], res["details"]["dt"], "put", 3)

    crr = {round(t, 4): s for t, s in front_crr}
    lsm = {round(t, 4): s for t, s in front_lsm}
    for t in (0.1, 0.5, 0.9):
        lignes.append({"grandeur": f"S* crr t={t}", "valeur": crr[round(t, 4)]})
        lignes.append({"grandeur": f"S* lsm t={t}", "valeur": lsm[round(t, 4)]})
    ecarts = [abs(lsm[t] - crr[t]) for t in lsm if t in crr and t <= 0.9 + 1e-9]
    lignes.append({"grandeur": "ecart max frontieres t<=0.9", "valeur": max(ecarts)})

    # attendre : interets perdus sur K, put europeen a S = 30
    lignes.append({"grandeur": "interets sur K (1 an)", "valeur": p["K"] * (exp(p["r"] * p["T"]) - 1)})
    lignes.append({"grandeur": "put europeen S=30", "valeur": prix_bs("put", 30.0, p["K"], p["T"], p["r"], p["sigma"])})

    # put totalenergies K = 86, lsm brut : ou et pourquoi on exerce
    K, T, n_dates = 86.0, 1.0, 50
    dt = T / n_dates
    S = simuler_trajectoires(cal["S0"], T, cal["r"], cal["sigma"], cal["q"], 100_000, n_dates,
                             Normales("box-muller", "park-miller", 42))
    flux, tau, _ = recurrence(S, K, cal["r"], dt, "put", 3, True)

    tot = tau < n_dates
    idx = np.flatnonzero(tot)
    S_ex = S[idx, tau[idx]]
    h = payoff("put", S_ex, K)
    eu = np.empty_like(h)
    for k in np.unique(tau[idx]):
        m = tau[idx] == k
        eu[m] = prix_bs_vect("put", S_ex[m], K, (n_dates - k) * dt, cal["r"], cal["sigma"], cal["q"])
    sous_eu = h < eu

    lignes.append({"grandeur": "TTE put prix brut", "valeur": float(flux.mean())})
    lignes.append({"grandeur": "TTE put part exercee", "valeur": float(tot.mean())})
    lignes.append({"grandeur": "TTE part des exercices apres t=0.6", "valeur": float(np.mean(tau[idx] * dt >= 0.6))})
    lignes.append({"grandeur": "TTE part des exercices sous l'europeen", "valeur": float(sous_eu.mean())})
    lignes.append({"grandeur": "TTE manque moyen (europeen - payoff)", "valeur": float(np.mean(eu[sous_eu] - h[sous_eu]))})
    lignes.append({"grandeur": "TTE part des mauvais exercices apres t=0.6",
                   "valeur": float(np.mean(tau[idx][sous_eu] * dt >= 0.6))})

    for l in lignes:
        print(f"  {l['grandeur']:45s} {l['valeur']:.4f}")
    ecrire_csv("comp_diagnostic", lignes)


# conditionnement de la regression au degre 5 : S brut contre x = S/K
def conditionnement():
    p = CAS_LS
    S = simuler_trajectoires(p["S0"], p["T"], p["r"], p["sigma"], p["q"], 100_000, 50,
                             Normales("box-muller", "park-miller", 42))
    s = S[:, 25]
    s = s[s < p["K"]]
    ligne = {"cond_S": float(np.linalg.cond(np.vander(s, 6, increasing=True))),
             "cond_x": float(np.linalg.cond(np.vander(s / p["K"], 6, increasing=True)))}
    print(ligne)
    ecrire_csv("comp_conditionnement", [ligne])


# validation croisee : meme option, deux codes ecrits separement (generateurs differents), 20 graines chacun
def validation_croisee():
    from validation import pricer_lsm as autre

    cas = {
        "put S0=36": ("put", dict(CAS_LS)),
        "put S0=40": ("put", dict(CAS_LS, S0=40.0)),
        "call S0=40 q=8%": ("call", dict(S0=40.0, K=40.0, T=1.0, r=0.06, sigma=0.2, q=0.08)),
    }
    lignes = []
    for nom, (type_option, p) in cas.items():
        nous = [prix_option_americaine(type_option, **p, graine=g)["prix"] for g in range(1, 21)]
        eux = [autre.prix_americaine_lsm(spot=p["S0"], strike=p["K"], volatilite=p["sigma"], taux=p["r"],
                                         maturite=p["T"], dividende=p["q"], type_option=type_option,
                                         degre=3, n_simulations=100_000, n_pas=50, graine=g)["prix"]
               for g in range(1, 21)]
        m1, s1 = float(np.mean(nous)), float(np.std(nous, ddof=1) / sqrt(20))
        m2, s2 = float(np.mean(eux)), float(np.std(eux, ddof=1) / sqrt(20))
        arbre_autre = autre.prix_arbre_binomial(spot=p["S0"], strike=p["K"], volatilite=p["sigma"], taux=p["r"],
                                                maturite=p["T"], dividende=p["q"], type_option=type_option,
                                                n_pas=5000, n_dates_exercice=50)
        lignes.append({"cas": nom, "nous": m1, "nous_se": s1, "autre": m2, "autre_se": s2,
                       "ecart_en_se": (m1 - m2) / sqrt(s1**2 + s2**2),
                       "arbre_nous": crr_bermudien(type_option, **p), "arbre_autre": arbre_autre})
        print(nom, round(m1, 4), round(m2, 4), round(lignes[-1]["ecart_en_se"], 2))
    ecrire_csv("comp_validation_croisee", lignes)


def main():
    print("Complements")
    validation_croisee()
    conditionnement()
    cal = calibrer()
    diagnostic(cal)
    plancher(cal)


if __name__ == "__main__":
    main()
