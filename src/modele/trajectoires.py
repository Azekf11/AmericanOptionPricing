from math import sqrt

import numpy as np


# trajectoires du mbg, schema exact :
# S(t+dt) = S(t) * exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z)
# sortie : matrice (n_simulations, n_pas + 1), colonne 0 = S0
def simuler_trajectoires(S0, T, r, sigma, q, n_simulations, n_pas, normales, antithetique=False):

    if n_simulations < 2 or n_pas < 1:
        raise ValueError("au moins 2 trajectoires et 1 pas")

    dt = T / n_pas
    derive = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)

    # tirages (antithetique : Z puis -Z)
    if antithetique:
        if n_simulations % 2:
            raise ValueError("nombre pair de trajectoires en antithetique")
        moitie = normales.tirer((n_simulations // 2) * n_pas).reshape(n_simulations // 2, n_pas)
        Z = np.vstack([moitie, -moitie])
    else:
        Z = normales.tirer(n_simulations * n_pas).reshape(n_simulations, n_pas)

    # cumul des log-rendements
    S = np.empty((n_simulations, n_pas + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(derive + diffusion * Z, axis=1))
    return S
