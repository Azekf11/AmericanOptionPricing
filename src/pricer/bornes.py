# borne haute duale (Andersen-Broadie, Haugh-Kogan, Rogers)
# pour toute martingale M nulle en 0 : prix <= E[ max_k (e^(-r t_k) h(S_k) - M_k) ]
# M est construite sur une fonction valeur approchee V_k = max(h, C_k),
# l'esperance a un pas E_k[V_(k+1)] est calculee par gauss-hermite (pas de simulation imbriquee)

import time
from math import exp, sqrt

import numpy as np

from src.aleatoire.normales import Normales
from src.modele.payoff import payoff
from src.modele.trajectoires import simuler_trajectoires
from src.pricer.longstaff_schwartz import Z_95, base


# N(x) vectorise, abramowitz-stegun 26.2.17 (erreur < 7.5e-8)
def N_rapide(x):
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    queue = np.exp(-0.5 * x**2) / sqrt(2 * np.pi) * poly
    return np.where(x >= 0, 1.0 - queue, queue)


# prix europeen sur un tableau de cours (echeance restante tau)
def europeen(type_option, S, K, tau, r, sigma, q):
    if tau <= 0:
        return payoff(type_option, S, K)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    if type_option == "call":
        return S * exp(-q * tau) * N_rapide(d1) - K * exp(-r * tau) * N_rapide(d2)
    return K * exp(-r * tau) * N_rapide(-d2) - S * exp(-q * tau) * N_rapide(-d1)


# base de la fonction valeur : 1, x, x^2, x^3 et le prix europeen (divise par K)
def base_valeur(type_option, S, K, tau, r, sigma, q):
    x = S / K
    return np.column_stack([np.ones_like(x), x, x**2, x**3,
                            europeen(type_option, S, K, tau, r, sigma, q) / K])


def borne_haute_duale(type_option, S0, K, T, r, sigma, q=0.0, n_dates=50, degre=3,
                      n_regression=100_000, n_exterieur=50_000, noeuds=40,
                      graine_regression=42, graine_exterieure=7):
    debut = time.perf_counter()
    N = max(1, int(round(n_dates * T)))
    dt = T / N

    # 1. regle LSM + regression de la fonction valeur (toutes les trajectoires)
    S = simuler_trajectoires(S0, T, r, sigma, q, n_regression, N,
                             Normales("box-muller", "park-miller", graine_regression))
    flux = payoff(type_option, S[:, N], K)
    tau = np.full(n_regression, N)
    beta = {}

    for k in range(N - 1, 0, -1):
        Y = flux * np.exp(-r * (tau - k) * dt)
        h = payoff(type_option, S[:, k], K)
        itm = h > 0

        # fonction valeur
        beta[k], *_ = np.linalg.lstsq(base_valeur(type_option, S[:, k], K, (N - k) * dt, r, sigma, q),
                                      Y, rcond=None)

        # regle d'exercice LSM (degre d, dans la monnaie)
        if itm.sum() > degre:
            b, *_ = np.linalg.lstsq(base(S[itm, k] / K, degre), Y[itm], rcond=None)
            ex = itm & (h > base(S[:, k] / K, degre) @ b)
            flux = np.where(ex, h, flux)
            tau = np.where(ex, k, tau)

    def V(S_k, k):
        h = payoff(type_option, S_k, K)
        if k == N:
            return h
        C = base_valeur(type_option, S_k.ravel(), K, (N - k) * dt, r, sigma, q) @ beta[k]
        return np.maximum(h, C.reshape(S_k.shape))

    # 2. martingale sur des trajectoires independantes
    z, w = np.polynomial.hermite_e.hermegauss(noeuds)
    w = w / w.sum()
    facteurs = np.exp((r - q - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * z)

    Se = simuler_trajectoires(S0, T, r, sigma, q, n_exterieur, N,
                              Normales("box-muller", "park-miller", graine_exterieure))
    M = np.zeros(n_exterieur)
    meilleur = payoff(type_option, Se[:, 0], K)

    for k in range(N):
        # esperance a un pas, exacte a la quadrature pres
        esperance = (V(Se[:, k:k + 1] * facteurs[None, :], k + 1) * w[None, :]).sum(axis=1)
        M = M + exp(-r * (k + 1) * dt) * (V(Se[:, k + 1], k + 1) - esperance)
        meilleur = np.maximum(meilleur, exp(-r * (k + 1) * dt) * payoff(type_option, Se[:, k + 1], K) - M)

    prix = float(meilleur.mean())
    erreur_type = float(meilleur.std(ddof=1) / sqrt(n_exterieur))
    return {"prix": prix, "erreur_type": erreur_type,
            "ic95": (prix - Z_95 * erreur_type, prix + Z_95 * erreur_type),
            "temps": time.perf_counter() - debut}
