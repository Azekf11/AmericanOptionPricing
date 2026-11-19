import time
from math import exp, sqrt

import numpy as np

from src.aleatoire.normales import METHODES, Normales
from src.modele.black_scholes import prix_bs, prix_bs_vect
from src.modele.payoff import payoff
from src.modele.trajectoires import simuler_trajectoires


Z_95 = 1.959963984540054


# base de regression 1, x, ..., x^d
def base(x, degre):
    return np.vander(x, degre + 1, increasing=True)


# polynomes de laguerre ponderes (article LS) : 1, e^(-x/2) L_0, ..., e^(-x/2) L_(d-1)
def base_laguerre(x, degre):
    e = np.exp(-x / 2)
    colonnes = [np.ones_like(x)]
    L_avant, L = np.zeros_like(x), np.ones_like(x)
    for n in range(degre):
        colonnes.append(e * L)
        L_avant, L = L, ((2 * n + 1 - x) * L - n * L_avant) / (n + 1)
    return np.column_stack(colonnes)


BASES = {"monomes": base, "laguerre": base_laguerre}


# exercice si h > 0 et h > continuation
# plancher : on refuse l'exercice si h <= prix europeen de l'echeance restante
def decision(h, continuation, S_k, reste, plancher):
    exercer = (h > 0) & (h > continuation)

    if plancher is not None and exercer.any():
        idx = np.flatnonzero(exercer)
        eu = plancher(S_k[idx], reste)
        exercer[idx[h[idx] <= eu]] = False

    return exercer


# recurrence retrograde (S : trajectoires, colonne 0 = S0)
def recurrence(S, K, r, dt, type_option, degre=3, itm=True, plancher=None, fbase=base):
    n_traj, n_col = S.shape
    N = n_col - 1

    # a maturite
    flux = payoff(type_option, S[:, N], K)
    tau = np.full(n_traj, N)
    coefs = {}

    for k in range(N - 1, 0, -1):
        h = payoff(type_option, S[:, k], K)
        sel = h > 0 if itm else np.ones(n_traj, dtype=bool)
        if sel.sum() <= degre:
            continue

        # flux futur actualise depuis sa date d'exercice
        Y = flux * np.exp(-r * (tau - k) * dt)

        # regression sur x = S/K
        x = S[:, k] / K
        beta, *_ = np.linalg.lstsq(fbase(x[sel], degre), Y[sel], rcond=None)
        coefs[k] = beta
        continuation = fbase(x, degre) @ beta

        # exercice
        ex = decision(h, continuation, S[:, k], (N - k) * dt, plancher)
        flux = np.where(ex, h, flux)
        tau = np.where(ex, k, tau)

    return flux * np.exp(-r * tau * dt), tau, coefs


# regle deja estimee appliquee a de nouvelles trajectoires (hors echantillon)
def appliquer(S, K, r, dt, type_option, coefs, degre, plancher=None, fbase=base):
    n_traj, n_col = S.shape
    N = n_col - 1

    flux = payoff(type_option, S[:, N], K)
    tau = np.full(n_traj, N)
    actif = np.ones(n_traj, dtype=bool)

    for k in range(1, N):
        if k not in coefs:
            continue
        h = payoff(type_option, S[:, k], K)
        continuation = fbase(S[:, k] / K, degre) @ coefs[k]
        ex = actif & decision(h, continuation, S[:, k], (N - k) * dt, plancher)
        flux = np.where(ex, h, flux)
        tau = np.where(ex, k, tau)
        actif &= ~ex

    return flux * np.exp(-r * tau * dt), tau


# frontiere d'exercice de la regle estimee : (t, S*) a chaque date
def frontiere(coefs, K, dt, type_option, degre, grille=None, fbase=base):
    if grille is None:
        grille = np.linspace(0.3 * K, 2.0 * K, 4001)

    h = payoff(type_option, grille, K)
    points = []

    for k in sorted(coefs):
        continuation = fbase(grille / K, degre) @ coefs[k]
        ex = (h > 0) & (h > continuation)
        if ex.any():
            critique = grille[ex].max() if type_option == "put" else grille[ex].min()
            points.append((k * dt, critique))

    return np.array(points)


# en antithetique, une observation = moyenne d'une paire
def par_paires(x, antithetique):
    if antithetique:
        moitie = x.size // 2
        return 0.5 * (x[:moitie] + x[moitie:])
    return x


# moyenne et erreur type
def stats(flux, antithetique=False):
    x = par_paires(flux, antithetique)
    return float(x.mean()), float(x.std(ddof=1) / sqrt(x.size))


# variable de controle : Y = A - b (E - prix exact), b = cov(A, E) / var(E)
def controle(flux, flux_eu, prix_eu, antithetique=False):
    A = par_paires(flux, antithetique)
    E = par_paires(flux_eu, antithetique)

    cov = np.cov(A, E, ddof=1)
    b = cov[0, 1] / cov[1, 1]
    rho = cov[0, 1] / sqrt(cov[0, 0] * cov[1, 1])

    Y = A - b * (E - prix_eu)
    return float(Y.mean()), float(Y.std(ddof=1) / sqrt(Y.size)), float(b), float(rho)


# pricer d'option americaine (fonction demandee par le sujet)
def prix_option_americaine(type_option, S0, K, T, r, sigma, q=0.0,
                           n_simulations=100_000, n_dates=50,
                           degre=3, itm_seulement=True,
                           methode_normale="box-muller", graine=42,
                           generateur_uniforme="park-miller",
                           antithetique=False, hors_echantillon=False,
                           controle_europeen=False, plancher_europeen=False,
                           base_regression="monomes"):

    # verification des entrees
    if type_option not in ("call", "put"):
        raise ValueError("type_option : 'call' ou 'put'")
    if min(S0, K, T, sigma) <= 0:
        raise ValueError("S0, K, T et sigma doivent etre > 0")
    if not (isinstance(degre, (int, np.integer)) and degre >= 1):
        raise ValueError("degre : entier >= 1")
    if n_simulations < 2 * (degre + 1):
        raise ValueError("pas assez de trajectoires pour ce degre")
    if methode_normale not in METHODES:
        raise ValueError("methode_normale inconnue")
    if generateur_uniforme not in ("park-miller", "wichmann-hill"):
        raise ValueError("generateur_uniforme inconnu")
    if base_regression not in BASES:
        raise ValueError("base_regression : 'monomes' ou 'laguerre'")
    if antithetique and n_simulations % 2:
        n_simulations += 1

    debut = time.perf_counter()

    # grille de dates (n_dates par an)
    N = max(1, int(round(n_dates * T)))
    dt = T / N
    normales = Normales(methode_normale, generateur_uniforme, graine)

    # plancher europeen (option)
    def europeen(S_noeuds, reste):
        return prix_bs_vect(type_option, S_noeuds, K, reste, r, sigma, q)

    plancher = europeen if plancher_europeen else None

    # simulation + recurrence
    S = simuler_trajectoires(S0, T, r, sigma, q, n_simulations, N, normales, antithetique)
    fbase = BASES[base_regression]
    flux, tau, coefs = recurrence(S, K, r, dt, type_option, degre, itm_seulement, plancher, fbase)

    # hors echantillon : nouveau jeu de trajectoires, meme regle
    if hors_echantillon:
        S = simuler_trajectoires(S0, T, r, sigma, q, n_simulations, N, normales, antithetique)
        flux, tau = appliquer(S, K, r, dt, type_option, coefs, degre, plancher, fbase)

    # europeenne sur les memes trajectoires
    flux_eu = payoff(type_option, S[:, -1], K) * exp(-r * T)
    prix_eu = prix_bs(type_option, S0, K, T, r, sigma, q)
    prix_vc, erreur_vc, b, rho = controle(flux, flux_eu, prix_eu, antithetique)

    details = {
        "n_dates": N,
        "dt": dt,
        "coefs": coefs,
        "part_exercee": float(np.mean(tau < N)),
        "prix_eu_mc": stats(flux_eu, antithetique)[0],
        "prix_eu_bs": prix_eu,
        "prime": prix_vc - prix_eu,
        "erreur_prime": erreur_vc,
        "b_controle": b,
        "correlation": rho,
    }

    # prix final
    if controle_europeen:
        prix, erreur_type = prix_vc, erreur_vc
    else:
        prix, erreur_type = stats(flux, antithetique)

    # exercice immediat en t = 0
    h0 = float(payoff(type_option, np.array([S0]), K)[0])
    if h0 > prix:
        prix, erreur_type = h0, 0.0

    temps = time.perf_counter() - debut
    ic95 = (prix - Z_95 * erreur_type, prix + Z_95 * erreur_type)

    return {"prix": prix, "erreur_type": erreur_type, "ic95": ic95, "temps": temps,
            "details": details}


# affichage court d'un resultat
def resume(res):
    bas, haut = res["ic95"]
    return (f"prix = {res['prix']:.4f}  (erreur type {res['erreur_type']:.4f}, "
            f"IC 95 % [{bas:.4f} ; {haut:.4f}], {res['temps']:.2f} s)")
