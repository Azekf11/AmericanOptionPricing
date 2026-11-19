# verifications en plus des trois questions
# lancer depuis la racine :  python -m questions.complements

from math import sqrt

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.calibration.totalenergies import calibrer
from src.modele.black_scholes import prix_bs
from src.pricer.longstaff_schwartz import prix_option_americaine


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
    cal = calibrer()
    plancher(cal)


if __name__ == "__main__":
    main()
