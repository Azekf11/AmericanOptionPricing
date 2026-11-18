from math import exp, sqrt

import numpy as np

from src.modele.payoff import payoff


# arbre de cox-ross-rubinstein
# exercice : "europeen", "americain" ou "bermudien" (n_dates_exercice dates)
def prix_crr(type_option, S0, K, T, r, sigma, q=0.0, n_pas=2000, exercice="americain",
             n_dates_exercice=None, frontiere=False):

    if exercice not in ("europeen", "americain", "bermudien"):
        raise ValueError("exercice inconnu : " + str(exercice))

    # u, d, p
    dt = T / n_pas
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("p hors de ]0, 1[ : augmenter n_pas")
    actu = exp(-r * dt)

    # pas entre deux dates d'exercice
    if exercice == "bermudien":
        if not n_dates_exercice or n_pas % n_dates_exercice:
            raise ValueError("n_pas doit etre un multiple de n_dates_exercice")
        ecart = n_pas // n_dates_exercice
    else:
        ecart = 1

    # payoff a maturite
    j = np.arange(n_pas + 1)
    valeurs = payoff(type_option, S0 * u ** (2 * j - n_pas), K)
    points = []

    # remontee de l'arbre
    for i in range(n_pas - 1, -1, -1):
        valeurs = actu * (p * valeurs[1:] + (1.0 - p) * valeurs[:-1])

        if exercice == "europeen" or i % ecart:
            continue

        # exercice anticipe
        S_noeuds = S0 * u ** (2 * np.arange(i + 1) - i)
        h = payoff(type_option, S_noeuds, K)
        exerce = (h > 0) & (h >= valeurs)
        valeurs = np.where(exerce, h, valeurs)

        # prix critique a cette date
        if frontiere and i > 0 and exerce.any():
            critique = S_noeuds[exerce].max() if type_option == "put" else S_noeuds[exerce].min()
            points.append((i * dt, critique))

    prix = float(valeurs[0])
    if frontiere:
        return prix, np.array(sorted(points))
    return prix
