# verifications en plus des trois questions
# lancer depuis la racine :  python -m questions.complements

from math import sqrt

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.pricer.longstaff_schwartz import prix_option_americaine


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


if __name__ == "__main__":
    main()
