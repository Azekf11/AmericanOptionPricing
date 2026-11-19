# question 1 : pricer call / put, effet du degre et des trajectoires retenues
# lancer depuis la racine :  python -m questions.q1_degre

import numpy as np

from questions.outils import CAS_LS, crr_americain, crr_bermudien, ecrire_csv
from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs
from src.pricer.longstaff_schwartz import prix_option_americaine, recurrence


GRAINES = range(1, 11)
N_SIM = 100_000

# exemple a 8 trajectoires du poly (K = 1,10, r = 6 %, 3 dates)
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


def exemple_huit():
    flux, tau, _ = recurrence(HUIT, K=1.10, r=0.06, dt=1.0, type_option="put", degre=2)
    europeen = np.maximum(1.10 - HUIT[:, -1], 0).mean() * np.exp(-0.06 * 3)
    print("8 trajectoires :", round(flux.mean(), 4), "dates d'exercice", tau.tolist())
    ecrire_csv("q1_exemple_8", [{"americain": flux.mean(), "europeen": europeen,
                                 "dates_exercice": " ".join(str(t) for t in tau)}])


def prix_de_base():
    lignes = []
    for type_option in ("put", "call"):
        res = prix_option_americaine(type_option, **CAS_LS)
        lignes.append({
            "option": type_option,
            "lsm": res["prix"],
            "erreur_type": res["erreur_type"],
            "ic_bas": res["ic95"][0],
            "ic_haut": res["ic95"][1],
            "temps": res["temps"],
            "europeen_bs": prix_bs(type_option, **CAS_LS),
            "crr_europeen": prix_crr(type_option, **CAS_LS, n_pas=2000, exercice="europeen"),
            "crr_bermudien": crr_bermudien(type_option, **CAS_LS),
            "crr_americain": crr_americain(type_option, **CAS_LS),
            "part_exercee": res["details"]["part_exercee"],
        })
        print(type_option, round(res["prix"], 4), "+/-", round(res["erreur_type"], 4))
    ecrire_csv("q1_prix_base", lignes)


# memes graines pour tous les reglages : les ecarts viennent du reglage, pas du tirage
# trois spots : dans la monnaie (36), a la monnaie (40), hors de la monnaie (44)
def degre_et_trajectoires():
    lignes = []

    for S0 in (36.0, 40.0, 44.0):
        p = dict(CAS_LS, S0=S0)
        ref = crr_bermudien("put", **p)
        for itm in (True, False):
            for degre in range(1, 6):
                prix, temps = [], []
                for g in GRAINES:
                    res = prix_option_americaine("put", **p, n_simulations=N_SIM,
                                                 degre=degre, itm_seulement=itm, graine=g)
                    prix.append(res["prix"])
                    temps.append(res["temps"])

                prix = np.array(prix)
                se = prix.std(ddof=1) / np.sqrt(prix.size)
                lignes.append({
                    "S0": S0,
                    "trajectoires": "itm" if itm else "toutes",
                    "degre": degre,
                    "prix_moyen": prix.mean(),
                    "se_moyenne": se,
                    "reference": ref,
                    "ecart_berm": prix.mean() - ref,
                    "ecart_en_se": (prix.mean() - ref) / se,
                    "temps_moyen": float(np.mean(temps)),
                })
                print(S0, lignes[-1]["trajectoires"], degre, round(lignes[-1]["ecart_berm"], 4))

    ecrire_csv("q1_degre", lignes)


# monomes contre polynomes de laguerre ponderes (base de l'article), memes graines
def bases():
    ref = crr_bermudien("put", **CAS_LS)
    lignes = []
    for nom in ("monomes", "laguerre"):
        for degre in (2, 3, 4):
            prix = np.array([prix_option_americaine("put", **CAS_LS, n_simulations=N_SIM, degre=degre,
                                                    base_regression=nom, graine=g)["prix"]
                             for g in GRAINES])
            se = prix.std(ddof=1) / np.sqrt(prix.size)
            lignes.append({"base": nom, "degre": degre, "prix_moyen": prix.mean(), "se_moyenne": se,
                           "ecart_berm": prix.mean() - ref, "ecart_en_se": (prix.mean() - ref) / se})
            print(nom, degre, round(prix.mean() - ref, 4))
    ecrire_csv("q1_bases", lignes)


def main():
    print("Q1")
    exemple_huit()
    prix_de_base()
    degre_et_trajectoires()
    bases()


if __name__ == "__main__":
    main()
