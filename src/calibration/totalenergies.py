import csv
from datetime import date
from math import exp, log, sqrt
from pathlib import Path

import numpy as np


DOSSIER = Path(__file__).resolve().parents[2] / "donnees"
FICHIER_COURS = DOSSIER / "TTE_PA_2024-11_2026-11.csv"
FICHIER_DIV = DOSSIER / "TTE_PA_dividendes.csv"

DATE_VALO = date(2026, 11, 10)     # derniere seance complete
ESTR = 0.02442                     # moyenne novembre 2026
EURIBOR_12M = 0.03204              # 09/11/2026, test de sensibilite
JOURS = 252


def lire_cours():
    dates, cloture, ajuste = [], [], []
    with open(FICHIER_COURS, newline="") as f:
        for ligne in csv.DictReader(f):
            dates.append(date.fromisoformat(ligne["Date"][:10]))
            cloture.append(float(ligne["Close"]))
            ajuste.append(float(ligne["Adj Close"]))
    return dates, np.array(cloture), np.array(ajuste)


def lire_dividendes():
    with open(FICHIER_DIV, newline="") as f:
        return [(date.fromisoformat(l["Date"]), float(l["Dividende"])) for l in csv.DictReader(f)]


# volatilite historique annualisee (rendements log quotidiens)
def volatilite(prix, n=JOURS):
    rend = np.diff(np.log(prix))[-n:]
    return float(rend.std(ddof=1) * sqrt(JOURS))


# parametres du modele a la date de valorisation
def calibrer(r=ESTR):
    dates, cloture, ajuste = lire_cours()
    fin = max(i for i, d in enumerate(dates) if d <= DATE_VALO)

    # spot et volatilite (cours ajustes des dividendes)
    S0 = float(cloture[fin])
    sigma = volatilite(ajuste[:fin + 1])
    sigma_brut = volatilite(cloture[:fin + 1])

    # dividendes des 12 derniers mois
    il_y_a_un_an = date(DATE_VALO.year - 1, DATE_VALO.month, DATE_VALO.day)
    derniers = [(d, m) for d, m in lire_dividendes() if il_y_a_un_an < d <= DATE_VALO]
    total = sum(m for _, m in derniers)

    # q continu tel que S0 e^(-qT) = S0 - VA(dividendes de l'annee a venir)
    va = sum(m * exp(-r * ((d.replace(year=d.year + 1) - DATE_VALO).days / 365.0))
             for d, m in derniers)
    q = -log(1.0 - va / S0)

    return {"S0": S0, "sigma": sigma, "sigma_brut": sigma_brut, "q": q, "r": r,
            "dividendes_12m": total, "rendement_simple": total / S0,
            "detail_dividendes": derniers, "date": DATE_VALO}


if __name__ == "__main__":
    p = calibrer()
    print(f"TotalEnergies au {p['date']} : S0 = {p['S0']:.2f} EUR")
    print(f"sigma = {p['sigma']:.2%} (cours bruts : {p['sigma_brut']:.2%})")
    print(f"dividendes 12 mois = {p['dividendes_12m']:.2f} EUR -> q = {p['q']:.2%}")
    print(f"r (ESTR) = {p['r']:.3%}")
