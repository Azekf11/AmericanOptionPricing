from math import erf, exp, log, sqrt

import numpy as np


# repartition N(0,1)
def N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


N_vect = np.vectorize(N, otypes=[float])


# prix europeen black-scholes (dividende continu q)
def prix_bs(type_option, S0, K, T, r, sigma, q=0.0):
    # d1, d2
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    # call
    if type_option == "call":
        return S0 * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)

    # put
    if type_option == "put":
        return K * exp(-r * T) * N(-d2) - S0 * exp(-q * T) * N(-d1)

    raise ValueError("type d'option inconnu : " + str(type_option))


# meme formule sur un tableau de prix S, echeance restante tau
def prix_bs_vect(type_option, S, K, tau, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)

    # d1, d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)

    if type_option == "call":
        return S * exp(-q * tau) * N_vect(d1) - K * exp(-r * tau) * N_vect(d2)
    return K * exp(-r * tau) * N_vect(-d2) - S * exp(-q * tau) * N_vect(-d1)
