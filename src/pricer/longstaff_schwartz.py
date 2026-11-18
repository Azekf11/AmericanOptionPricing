import time
from math import exp, sqrt

import numpy as np

from src.aleatoire.normales import METHODES, Normales
from src.modele.black_scholes import prix_bs
from src.modele.payoff import payoff
from src.modele.trajectoires import simuler_trajectoires


Z_95 = 1.959963984540054


# base de regression 1, x, ..., x^d
def base(x, degre):
    return np.vander(x, degre + 1, increasing=True)


# exercice si h > 0 et h > continuation
def decision(h, continuation):
    return (h > 0) & (h > continuation)


# recurrence retrograde (S : trajectoires, colonne 0 = S0)
def recurrence(S, K, r, dt, type_option, degre=3, itm=True):
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
        beta, *_ = np.linalg.lstsq(base(x[sel], degre), Y[sel], rcond=None)
        coefs[k] = beta
        continuation = base(x, degre) @ beta

        # exercice
        ex = decision(h, continuation)
        flux = np.where(ex, h, flux)
        tau = np.where(ex, k, tau)

    return flux * np.exp(-r * tau * dt), tau, coefs


# moyenne et erreur type
def stats(flux):
    return float(flux.mean()), float(flux.std(ddof=1) / sqrt(flux.size))


# pricer d'option americaine (fonction demandee par le sujet)
def prix_option_americaine(type_option, S0, K, T, r, sigma, q=0.0,
                           n_simulations=100_000, n_dates=50,
                           degre=3, itm_seulement=True,
                           methode_normale="box-muller", graine=42,
                           generateur_uniforme="park-miller"):

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

    debut = time.perf_counter()

    # grille de dates (n_dates par an)
    N = max(1, int(round(n_dates * T)))
    dt = T / N
    normales = Normales(methode_normale, generateur_uniforme, graine)

    # simulation + recurrence
    S = simuler_trajectoires(S0, T, r, sigma, q, n_simulations, N, normales)
    flux, tau, coefs = recurrence(S, K, r, dt, type_option, degre, itm_seulement)

    # europeenne sur les memes trajectoires
    flux_eu = payoff(type_option, S[:, -1], K) * exp(-r * T)
    prix_eu = prix_bs(type_option, S0, K, T, r, sigma, q)

    details = {
        "n_dates": N,
        "dt": dt,
        "coefs": coefs,
        "part_exercee": float(np.mean(tau < N)),
        "prix_eu_mc": stats(flux_eu)[0],
        "prix_eu_bs": prix_eu,
    }

    # prix final
    prix, erreur_type = stats(flux)

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
