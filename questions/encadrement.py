# encadrement du prix du put LS : borne basse, borne haute duale, extrapolation de richardson
# lancer depuis la racine :  python -m questions.encadrement

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.modele.arbre_crr import prix_crr
from src.pricer.bornes import borne_haute_duale
from src.pricer.longstaff_schwartz import prix_option_americaine


GRAINES = range(1, 11)
N_SIM = 200_000


def moyenne(valeurs):
    v = np.array(valeurs)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


# richardson : le prix bermudien a n dates converge en 1/n, donc A ~ 2 B(2n) - B(n)
def richardson():
    lignes = []
    for n in (25, 50, 100, 200):
        berm = prix_crr("put", **CAS_LS, n_pas=40 * n, exercice="bermudien", n_dates_exercice=n)
        if n <= 100:
            lsm = [prix_option_americaine("put", **CAS_LS, n_simulations=N_SIM, n_dates=n, graine=g,
                                          antithetique=True, controle_europeen=True)["prix"]
                   for g in GRAINES]
            m, s = moyenne(lsm)
        else:
            m, s = float("nan"), float("nan")
        lignes.append({"dates": n, "crr_bermudien": berm, "lsm": m, "lsm_se": s})
        print("dates", n, round(berm, 4), round(m, 4))
    ecrire_csv("enc_richardson", lignes)
    return {l["dates"]: l for l in lignes}


def main():
    print("Encadrement")
    rich = richardson()
    americain = prix_crr("put", **CAS_LS, n_pas=8000)

    # borne basse : regle estimee puis appliquee a des trajectoires neuves
    bas = moyenne([prix_option_americaine("put", **CAS_LS, n_simulations=N_SIM, graine=g,
                                          antithetique=True, hors_echantillon=True)["prix"]
                   for g in GRAINES])

    # borne haute duale (40 noeuds de gauss-hermite, puis 80 pour verifier la quadrature)
    haut = borne_haute_duale("put", **CAS_LS)
    haut_80 = borne_haute_duale("put", **CAS_LS, noeuds=80)

    lsm_50 = (rich[50]["lsm"], rich[50]["lsm_se"])
    rich_lsm = 2 * rich[100]["lsm"] - rich[50]["lsm"]
    rich_lsm_se = np.sqrt(4 * rich[100]["lsm_se"] ** 2 + rich[50]["lsm_se"] ** 2)
    rich_crr = 2 * rich[100]["crr_bermudien"] - rich[50]["crr_bermudien"]

    lignes = [
        {"estimateur": "borne basse (hors echantillon)", "prix": bas[0], "erreur_type": bas[1], "reference": crr_bermudien("put", **CAS_LS)},
        {"estimateur": "LSM in-sample", "prix": lsm_50[0], "erreur_type": lsm_50[1], "reference": crr_bermudien("put", **CAS_LS)},
        {"estimateur": "borne haute duale", "prix": haut["prix"], "erreur_type": haut["erreur_type"], "reference": crr_bermudien("put", **CAS_LS)},
        {"estimateur": "borne haute duale 80 noeuds", "prix": haut_80["prix"], "erreur_type": haut_80["erreur_type"], "reference": crr_bermudien("put", **CAS_LS)},
        {"estimateur": "richardson arbre", "prix": rich_crr, "erreur_type": 0.0, "reference": americain},
        {"estimateur": "richardson LSM", "prix": rich_lsm, "erreur_type": rich_lsm_se, "reference": americain},
    ]
    for l in lignes:
        print(f"  {l['estimateur']:32s} {l['prix']:.4f} +/- {l['erreur_type']:.4f}")
    ecrire_csv("enc_bornes", lignes)


if __name__ == "__main__":
    main()
