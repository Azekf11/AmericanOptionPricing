# figures (largeur d'une colonne, deux panneaux l'un sous l'autre)
# fig 1 : la decision d'exercice (regression a t = 0,5 puis trajectoires)
# fig 2 : question 3 (dispersion des prix, cout de la precision)
# fig 3 : encadrement du prix (bornes, richardson)
# lancer depuis la racine :  python -m questions.figures

import numpy as np
from matplotlib.ticker import NullFormatter

from questions.outils import (BLEU, BRIQUE, CAS_LS, GRIS, GRIS_CLAIR, NOIR, VERT, crr_bermudien,
                              ecrire_csv, enregistrer, fr, lettre, lire_csv, style, virgules)
from src.aleatoire.normales import Normales
from src.modele.arbre_crr import prix_crr
from src.modele.trajectoires import simuler_trajectoires
from src.pricer.longstaff_schwartz import base, frontiere, recurrence


N_SIM = 100_000
N_DATES = 50
DATE_COUPE = 25          # t = 0,5
TAILLE = (3.45, 2.95)    # pouces, largeur d'une colonne


def simulation_ls():
    p = CAS_LS
    S = simuler_trajectoires(p["S0"], p["T"], p["r"], p["sigma"], p["q"], N_SIM, N_DATES,
                             Normales("box-muller", "park-miller", 42))
    return S, p["T"] / N_DATES


# fig 1 (a) : ce que rapporte l'attente a t = 0,5
# gain = continuation estimee - exercice immediat
def panneau_regression(ax, S, dt):
    K, r = CAS_LS["K"], CAS_LS["r"]
    k = DATE_COUPE

    # flux futurs actualises a la date k (on remonte de T jusqu'a k + 1)
    Y, _, _ = recurrence(S[:, k:], K, r, dt, "put", degre=3, itm=True)
    Sk = S[:, k]
    itm = Sk < K

    ax.axhline(0, color=NOIR, lw=0.7)

    # nuage (1 500 premieres trajectoires dans la monnaie)
    idx = np.flatnonzero(itm)[:1500]
    ax.scatter(Sk[idx], Y[idx] - (K - Sk[idx]), s=2, color=GRIS_CLAIR, lw=0, rasterized=True,
               label="trajectoire")

    # regressions de degre 1, 2, 3 sur toutes les trajectoires dans la monnaie
    grille = np.linspace(26.0, K, 1401)
    styles = {1: (VERT, (0, (4, 2))), 2: (BRIQUE, (0, (1.2, 1.2))), 3: (BLEU, "-")}
    seuils = {}
    for d, (couleur, trait) in styles.items():
        beta, *_ = np.linalg.lstsq(base(Sk[itm] / K, d), Y[itm], rcond=None)
        gain = base(grille / K, d) @ beta - (K - grille)
        seuils[d] = grille[gain < 0].max()
        ax.plot(grille, gain, color=couleur, ls=trait, lw=1.3, label=f"degré {d}")

    # prix critique (degre 3) : le gain change de signe
    s3 = seuils[3]
    ax.plot([s3], [0], ls="none", marker="o", ms=3.5, mfc="white", mec=NOIR, mew=0.8, zorder=5)
    ax.annotate(f"$S^*$ ≈ {fr(s3, 1)}", xy=(s3, 0), xytext=(s3 - 4.6, -1.8),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=NOIR))
    ax.text(26.6, -2.9, "exercer", fontsize=7, style="italic")
    ax.text(39.6, 2.75, "attendre", fontsize=7, style="italic", ha="right")

    ax.set_xlim(26, K)
    ax.set_ylim(-3.4, 3.4)
    ax.set_xlabel("cours $S$ à $t$ = 0,5", labelpad=1)
    ax.set_ylabel("gain à attendre (€)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=4, markerscale=3,
              columnspacing=0.9, handlelength=1.9, handletextpad=0.4)
    virgules(ax)
    return seuils


# fig 1 (b) : quelques trajectoires, ou elles s'arretent, et la frontiere
def panneau_trajectoires(ax, S, dt, coefs):
    K, r = CAS_LS["K"], CAS_LS["r"]
    t = np.arange(N_DATES + 1) * dt

    # date d'exercice de chaque trajectoire (regle estimee sur les 100 000)
    _, tau, _ = recurrence(S, K, r, dt, "put", degre=3, itm=True)

    # 6 trajectoires exercees avant T, 4 gardees jusqu'au bout
    exercees = np.flatnonzero(tau < N_DATES)[:6]
    gardees = np.flatnonzero(tau == N_DATES)[:4]
    for i in np.concatenate([exercees, gardees]):
        ax.plot(t, S[i], color=GRIS_CLAIR, lw=0.55)
        ax.plot(t[:tau[i] + 1], S[i, :tau[i] + 1], color=GRIS, lw=0.65)
    ax.plot(t[tau[exercees]], S[exercees, tau[exercees]], ls="none", marker="o", ms=3.2,
            mfc=BRIQUE, mec="white", mew=0.5, label="exercice", zorder=4)

    # frontieres
    _, crr = prix_crr("put", **CAS_LS, n_pas=2000, exercice="bermudien", n_dates_exercice=N_DATES,
                      frontiere=True)
    lsm = frontiere(coefs, K, dt, "put", 3)
    ax.plot(crr[:, 0], crr[:, 1], color=NOIR, lw=1.1, label="$S^*(t)$ arbre")
    ax.plot(lsm[:, 0], lsm[:, 1], ls="none", marker="s", ms=1.8, color=BLEU, label="$S^*(t)$ LSM")
    ax.axhline(K, color=GRIS, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.99, K + 0.5, "$K$ = 40", fontsize=6.5, color=GRIS, ha="right")

    ax.set_xlim(0, 1)
    ax.set_ylim(27, 48)
    ax.set_xlabel("temps $t$ (années)", labelpad=1)
    ax.set_ylabel("cours $S_t$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, columnspacing=1.0,
              handletextpad=0.4)
    virgules(ax)


def figure_1():
    S, dt = simulation_ls()
    _, _, coefs = recurrence(S, CAS_LS["K"], CAS_LS["r"], dt, "put", degre=3, itm=True)

    plt = style()
    fig, (a, b) = plt.subplots(2, 1, figsize=TAILLE)
    seuils = panneau_regression(a, S, dt)
    panneau_trajectoires(b, S, dt, coefs)
    lettre(a, "(a)")
    lettre(b, "(b)")
    fig.tight_layout(h_pad=1.6)
    enregistrer(fig, "fig1_decision")

    ecrire_csv("fig1_seuils", [{"degre": d, "S_etoile_t05": s} for d, s in seuils.items()])


# fig 2 (a) : prix des 20 graines pour chaque N
def panneau_boites(ax, par_graine, ref):
    tailles = sorted({int(l["N"]) for l in par_graine})
    donnees = [[l["prix"] for l in par_graine if l["estimateur"] == "brut" and l["N"] == N]
               for N in tailles]
    pos = np.arange(len(tailles))

    ax.axhline(ref, color=NOIR, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(len(tailles) - 0.6, ref + 0.015, f"arbre CRR : {fr(ref, 3)}", fontsize=6.5,
            ha="right", va="bottom")
    ax.boxplot(donnees, positions=pos, widths=0.5, patch_artist=True, whis=(0, 100),
               boxprops=dict(facecolor="#dce6f2", edgecolor=BLEU, lw=0.8),
               medianprops=dict(color=BLEU, lw=1.3),
               whiskerprops=dict(color=BLEU, lw=0.8), capprops=dict(color=BLEU, lw=0.8))

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{N:,}".replace(",", " ") for N in tailles])
    ax.set_xlabel("nombre de trajectoires $N$", labelpad=1)
    ax.set_ylabel("prix du put")
    virgules(ax, "y", dec=1)


# fig 2 (b) : demi-largeur de l'ic contre le temps de calcul
def panneau_cout(ax, convergence, resume):
    noms = {"brut": "brut", "antithetique": "antithétique", "controle": "contrôle"}
    styles = {"brut": (BLEU, "o", "-"), "antithetique": (BRIQUE, "s", "--"),
              "controle": (VERT, "^", "-.")}
    pentes = {l["estimateur"]: l["pente_ic"] for l in resume}

    for nom, (couleur, marque, trait) in styles.items():
        sel = [l for l in convergence if l["estimateur"] == nom]
        ax.plot([l["temps"] for l in sel], [l["demi_ic95"] for l in sel], color=couleur,
                marker=marque, ms=3, ls=trait, lw=1.0,
                label=f"{noms[nom]} ({fr(pentes[nom], 3)})")

    # guide de pente -1/2
    sel = [l for l in convergence if l["estimateur"] == "brut"]
    t = np.array([l["temps"] for l in sel])
    guide = 2.0 * sel[0]["demi_ic95"] * (t / t[0]) ** -0.5
    ax.plot(t, guide, color=GRIS, lw=0.7, ls=":")
    ax.text(t[2], guide[2] * 1.25, "pente $-1/2$", fontsize=6.5, color=GRIS, rotation=-17)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("temps de calcul d'un prix (s)", labelpad=1)
    ax.set_ylabel("demi-largeur IC 95 % (€)")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    virgules(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, columnspacing=1.0,
              handlelength=1.9, handletextpad=0.4)


def figure_2():
    par_graine = lire_csv("q3_par_graine")
    convergence = lire_csv("q3_convergence")
    resume = lire_csv("q3_resume")
    ref = crr_bermudien("put", **CAS_LS)

    plt = style()
    fig, (a, b) = plt.subplots(2, 1, figsize=TAILLE)
    panneau_boites(a, par_graine, ref)
    panneau_cout(b, convergence, resume)
    lettre(a, "(a)")
    lettre(b, "(b)")
    fig.tight_layout(h_pad=1.6)
    enregistrer(fig, "fig2_convergence")


# fig 3 : tous les estimateurs du put sur une meme echelle
def figure_3():
    bornes = {l["estimateur"]: l for l in lire_csv("enc_bornes")}
    rich = {int(l["dates"]): l for l in lire_csv("enc_richardson")}
    bermudien = rich[50]["crr_bermudien"]
    americain = bornes["richardson arbre"]["reference"]

    lignes = [
        ("borne basse (hors échantillon)", bornes["borne basse (hors echantillon)"], BLEU, "o"),
        ("LSM in-sample", bornes["LSM in-sample"], GRIS, "o"),
        ("borne haute duale", bornes["borne haute duale"], BRIQUE, "s"),
        ("Richardson sur le LSM", bornes["richardson LSM"], VERT, "D"),
    ]

    plt = style()
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    bas, haut = lignes[0][1]["prix"], lignes[2][1]["prix"]
    ax.axvspan(bas, haut, color=GRIS_CLAIR, alpha=0.45, lw=0)
    ax.axvline(bermudien, color=NOIR, lw=0.9, ls=(0, (4, 2)))
    ax.axvline(americain, color=NOIR, lw=0.9, ls=":")
    ax.text(bermudien - 0.0002, len(lignes) - 0.45, "bermudienne exacte", fontsize=6, ha="right", va="bottom")
    ax.text(americain + 0.0002, len(lignes) - 0.45, "américaine", fontsize=6, ha="left", va="bottom")

    for i, (nom, l, couleur, marque) in enumerate(lignes):
        ax.errorbar(l["prix"], len(lignes) - 1 - i, xerr=1.96 * l["erreur_type"], fmt=marque, color=couleur,
                    ms=3.5, capsize=2, lw=1)

    ax.set_yticks(range(len(lignes)))
    ax.set_yticklabels([l[0] for l in lignes][::-1])
    ax.set_ylim(-0.6, len(lignes) + 0.35)
    ax.set_xlim(4.468, 4.490)
    ax.set_xticks([4.470, 4.475, 4.480, 4.485, 4.490])
    ax.set_xlabel("prix du put (cas de l'article), IC à 95 %", labelpad=1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    virgules(ax, "x", dec=3)
    enregistrer(fig, "fig3_encadrement")


def main():
    print("Figures")
    figure_1()
    figure_2()
    figure_3()


if __name__ == "__main__":
    main()
