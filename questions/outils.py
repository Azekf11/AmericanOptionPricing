import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.modele.arbre_crr import prix_crr


RACINE = Path(__file__).resolve().parents[1]
TABLEAUX = RACINE / "resultats" / "tableaux"
FIGURES = RACINE / "resultats" / "figures"

# cas de reference de longstaff-schwartz (2001), tableau 1
CAS_LS = dict(S0=36.0, K=40.0, T=1.0, r=0.06, sigma=0.20, q=0.0)
DATES_PAR_AN = 50

# couleurs (verifiees daltonisme) + gris
BLEU = "#2b62a8"
BRIQUE = "#d1603d"
VERT = "#2a9d8f"
NOIR = "#1a1a1a"
GRIS = "#6b6b6b"
GRIS_CLAIR = "#c9c9c9"


# csv

def ecrire_csv(nom, lignes):
    TABLEAUX.mkdir(parents=True, exist_ok=True)
    with open(TABLEAUX / f"{nom}.csv", "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.DictWriter(f, fieldnames=list(lignes[0]))
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"  -> resultats/tableaux/{nom}.csv")


def lire_csv(nom):
    def convertir(v):
        try:
            return float(v)
        except ValueError:
            return v

    with open(TABLEAUX / f"{nom}.csv", newline="", encoding="utf-8") as f:
        return [{k: convertir(v) for k, v in ligne.items()} for ligne in csv.DictReader(f)]


# references (arbre crr, meme grille de dates que le monte-carlo)

def crr_bermudien(type_option, S0, K, T, r, sigma, q=0.0):
    n_dates = int(round(DATES_PAR_AN * T))
    return prix_crr(type_option, S0, K, T, r, sigma, q, n_pas=40 * n_dates,
                    exercice="bermudien", n_dates_exercice=n_dates)


def crr_americain(type_option, S0, K, T, r, sigma, q=0.0):
    return prix_crr(type_option, S0, K, T, r, sigma, q, n_pas=int(round(2000 * T)))


# graphes

def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": NOIR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


# nombre a la francaise : 4,478 et vrai signe moins
def fr(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",").replace("-", "−")


# virgule decimale sur les axes
def virgules(ax, axes="xy", dec=None):
    def formater(x, _):
        if dec is None:
            texte = f"{x:g}"
        else:
            texte = f"{x:.{dec}f}"
        return texte.replace(".", ",").replace("-", "−")

    if "x" in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(formater))
    if "y" in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(formater))


# (a), (b) sous le titre de l'axe x
def lettre(ax, texte):
    ax.set_xlabel(ax.get_xlabel() + "\n" + texte)


def enregistrer(fig, nom):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{nom}.{ext}")
    plt.close(fig)
    print(f"  -> resultats/figures/{nom}.png")
