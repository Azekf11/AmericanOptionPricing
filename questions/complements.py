# verifications en plus des trois questions
# lancer depuis la racine :  python -m questions.complements

import time
from math import erf, exp, sqrt

import numpy as np

from questions.outils import CAS_LS, crr_bermudien, ecrire_csv
from src.aleatoire.lcg import A_PM, M_PM, LCGBlocs
from src.aleatoire.normales import METHODES, Normales
from src.calibration.totalenergies import calibrer
from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs, prix_bs_vect
from src.modele.payoff import payoff
from src.modele.trajectoires import simuler_trajectoires
from src.pricer.longstaff_schwartz import frontiere, prix_option_americaine, recurrence


GENERATEURS = ("park-miller", "wichmann-hill")


def phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# qualite des generateurs : 2 generateurs x 3 methodes
def generateurs(ref):
    n = 1_000_000
    queue_theorique = n * 2 * (1 - phi(3.5))
    lignes = []

    for gen in GENERATEURS:
        for methode in METHODES:
            asym, kurt, queues, temps = [], [], [], []
            for g in range(1, 21):
                debut = time.perf_counter()
                z = Normales(methode, gen, g).tirer(n)
                temps.append(time.perf_counter() - debut)
                asym.append(np.mean(z**3) / sqrt(6 / n))
                kurt.append((np.mean(z**4) - 3) / sqrt(24 / n))
                queues.append(np.sum(np.abs(z) > 3.5))

            prix = [prix_option_americaine("put", **CAS_LS, methode_normale=methode,
                                           generateur_uniforme=gen, graine=g)["prix"]
                    for g in range(1, 11)]

            lignes.append({
                "generateur": gen,
                "methode": methode,
                "asymetrie_z": float(np.mean(asym)),
                "kurtosis_z": float(np.mean(kurt)),
                "queues_obs": float(np.mean(queues)),
                "queues_theo": queue_theorique,
                "temps_1M": float(np.mean(temps)),
                "prix_moyen": float(np.mean(prix)),
                "se_moyenne": float(np.std(prix, ddof=1) / sqrt(len(prix))),
                "ecart_berm": float(np.mean(prix)) - ref,
            })
            print(gen, methode, round(lignes[-1]["prix_moyen"], 4))

    ecrire_csv("comp_generateurs", lignes)


# piege des graines d'un lcg (x0 = graine) contre des blocs disjoints
def graines():
    n = 1_000_000
    lignes = []

    for semis in ("x0 = graine", "blocs disjoints"):
        u, z = [], []
        for g in (1, 2):
            for liste, sortie in ((u, "u"), (z, "z")):
                gen = Normales("box-muller", "park-miller", g)
                if semis == "x0 = graine":
                    gen.u.lcg = [LCGBlocs(A_PM, M_PM, g)]
                liste.append(gen.u.tirer(n) if sortie == "u" else gen.tirer(n))

        lignes.append({"semis": semis,
                       "corr_u": float(np.corrcoef(u[0], u[1])[0, 1]),
                       "corr_abs_z": float(np.corrcoef(np.abs(z[0]), np.abs(z[1]))[0, 1])})
        print(lignes[-1])

    ecrire_csv("comp_graines", lignes)


# l'erreur type annoncee est-elle la vraie dispersion ? (60 graines)
def validation_erreur_type():
    n_sim = 10_000
    p = CAS_LS
    res = {"europeen": ([], []), "in-sample": ([], []), "hors echantillon": ([], [])}

    for g in range(1, 61):
        a = prix_option_americaine("put", **p, n_simulations=n_sim, graine=g)
        res["in-sample"][0].append(a["prix"])
        res["in-sample"][1].append(a["erreur_type"])

        S = simuler_trajectoires(p["S0"], p["T"], p["r"], p["sigma"], p["q"], n_sim, 50,
                                 Normales("box-muller", "park-miller", g))
        e = payoff("put", S[:, -1], p["K"]) * exp(-p["r"] * p["T"])
        res["europeen"][0].append(e.mean())
        res["europeen"][1].append(e.std(ddof=1) / sqrt(n_sim))

        h = prix_option_americaine("put", **p, n_simulations=n_sim, graine=g, hors_echantillon=True)
        res["hors echantillon"][0].append(h["prix"])
        res["hors echantillon"][1].append(h["erreur_type"])

    lignes = []
    for nom, (prix, se) in res.items():
        disp = float(np.std(prix, ddof=1))
        lignes.append({"estimateur": nom, "runs": len(prix), "N": n_sim,
                       "prix_moyen": float(np.mean(prix)), "dispersion": disp,
                       "erreur_type_annoncee": float(np.mean(se)),
                       "rapport": disp / float(np.mean(se)),
                       "incertitude_rapport": 1 / sqrt(2 * (len(prix) - 1))})
        print(nom, round(lignes[-1]["rapport"], 3))

    ecrire_csv("comp_erreur_type", lignes)


# biais in-sample / hors echantillon
def in_sample_hors_echantillon(ref):
    lignes = []
    for N in (10_000, 100_000):
        for mode, options in (("in-sample", {}), ("hors echantillon", {"hors_echantillon": True})):
            prix = [prix_option_americaine("put", **CAS_LS, n_simulations=N, graine=g, **options)["prix"]
                    for g in range(1, 21)]
            m = float(np.mean(prix))
            s = float(np.std(prix, ddof=1) / sqrt(len(prix)))
            lignes.append({"N": N, "estimateur": mode, "prix_moyen": m, "se_moyenne": s,
                           "ecart_berm": m - ref, "ecart_en_se": (m - ref) / s})
            print(N, mode, round(m, 4))

    ecrire_csv("comp_hors_echantillon", lignes)


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
    ref = crr_bermudien("put", **CAS_LS)
    cal = calibrer()
    diagnostic(cal)
    graines()
    validation_erreur_type()
    in_sample_hors_echantillon(ref)
    plancher(cal)
    generateurs(ref)


if __name__ == "__main__":
    main()
