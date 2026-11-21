# question 3 : impact du nombre de simulations sur l'intervalle de confiance
# put du cas LS, 20 graines par taille N, trois estimateurs
# + couverture sur 100 a 300 repetitions, in-sample contre hors echantillon
# lancer depuis la racine :  python -m questions.q3_convergence

from math import sqrt

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.aleatoire.normales import Normales
from src.modele.trajectoires import simuler_trajectoires
from src.pricer.longstaff_schwartz import Z_95, appliquer, prix_option_americaine, recurrence


TAILLES = (1_000, 5_000, 10_000, 50_000, 100_000, 200_000)
GRAINES = range(1, 21)
Z_99 = 2.5758293035489004

ESTIMATEURS = {
    "brut": {},
    "antithetique": {"antithetique": True},
    "controle": {"controle_europeen": True},
}


# pente en log-log
def pente(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    print("Q3")
    ref = crr_bermudien("put", **CAS_LS)

    # tous les prix, graine par graine
    brut = []
    for nom, options in ESTIMATEURS.items():
        for N in TAILLES:
            for g in GRAINES:
                res = prix_option_americaine("put", **CAS_LS, n_simulations=N, graine=g, **options)
                brut.append({"estimateur": nom, "N": N, "graine": g, "prix": res["prix"],
                             "erreur_type": res["erreur_type"], "temps": res["temps"]})
            print(nom, N)
    ecrire_csv("q3_par_graine", brut)

    # resume par (estimateur, N)
    lignes = []
    for nom in ESTIMATEURS:
        for N in TAILLES:
            sel = [b for b in brut if b["estimateur"] == nom and b["N"] == N]
            prix = np.array([b["prix"] for b in sel])
            se = np.array([b["erreur_type"] for b in sel])
            se_moy = prix.std(ddof=1) / np.sqrt(prix.size)

            lignes.append({
                "estimateur": nom,
                "N": N,
                "prix_moyen": prix.mean(),
                "ecart_berm": prix.mean() - ref,
                "ecart_en_se": (prix.mean() - ref) / se_moy,
                "erreur_type": se.mean(),
                "dispersion": prix.std(ddof=1),
                "demi_ic95": Z_95 * se.mean(),
                "demi_ic95_racine_N": Z_95 * se.mean() * sqrt(N),
                "demi_ic99": Z_99 * se.mean(),
                "couverture": float(np.mean(np.abs(prix - ref) <= Z_95 * se)),
                "temps": float(np.mean([b["temps"] for b in sel])),
            })
    ecrire_csv("q3_convergence", lignes)

    # pentes et efficacite a N = 100 000
    resume = []
    base = next(l for l in lignes if l["estimateur"] == "brut" and l["N"] == 100_000)
    for nom in ESTIMATEURS:
        sel = [l for l in lignes if l["estimateur"] == nom]
        l100 = next(l for l in sel if l["N"] == 100_000)
        resume.append({
            "estimateur": nom,
            "pente_ic": pente([l["N"] for l in sel], [l["demi_ic95"] for l in sel]),
            "pente_temps": pente([l["N"] for l in sel[2:]], [l["temps"] for l in sel[2:]]),
            "division_variance": (base["erreur_type"] / l100["erreur_type"]) ** 2,
            "rapport_temps": l100["temps"] / base["temps"],
            "reference": ref,
        })
        print(resume[-1])
    ecrire_csv("q3_resume", resume)

    couverture()


# un prix a partir d'une suite deja entamee (les repetitions ne se recouvrent pas)
def un_prix(normales, N, hors_echantillon):
    p = CAS_LS
    dt = p["T"] / 50
    S = simuler_trajectoires(p["S0"], p["T"], p["r"], p["sigma"], p["q"], N, 50, normales)
    flux, _, coefs = recurrence(S, p["K"], p["r"], dt, "put", degre=3, itm=True)
    if hors_echantillon:
        S = simuler_trajectoires(p["S0"], p["T"], p["r"], p["sigma"], p["q"], N, 50, normales)
        flux, _ = appliquer(S, p["K"], p["r"], dt, "put", coefs, 3)
    return flux.mean(), flux.std(ddof=1) / sqrt(N)


# l'IC a 95 % contient-il la reference 95 fois sur 100 ?
# une seule suite wichmann-hill (blocs de 2^36 valeurs), decoupee en morceaux consecutifs
def couverture(plans=((1_000, 300), (10_000, 300), (100_000, 100))):
    ref = crr_bermudien("put", **CAS_LS)
    normales = Normales("box-muller", "wichmann-hill", 1)
    lignes = []

    for N, repetitions in plans:
        for mode in ("in-sample", "hors echantillon"):
            res = np.array([un_prix(normales, N, mode == "hors echantillon") for _ in range(repetitions)])
            prix, se = res[:, 0], res[:, 1]
            se_moy = prix.std(ddof=1) / sqrt(repetitions)
            lignes.append({
                "N": N,
                "repetitions": repetitions,
                "estimateur": mode,
                "prix_moyen": prix.mean(),
                "biais": prix.mean() - ref,
                "biais_en_se": (prix.mean() - ref) / se_moy,
                "dispersion": prix.std(ddof=1),
                "erreur_type_moyenne": se.mean(),
                "couverture": float(np.mean(np.abs(prix - ref) <= Z_95 * se)),
            })
            print(N, mode, round(lignes[-1]["biais"], 4), round(lignes[-1]["couverture"], 3))

    ecrire_csv("q3_couverture", lignes)


if __name__ == "__main__":
    main()
