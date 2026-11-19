# second pricer LSM, ecrit separement du code de src/ pour verifier nos prix
# autre generateur (LCG modulo 2^64, constantes MMIX de Knuth), autre arbre, autre structure
# aucun tirage aleatoire de bibliotheque : LCG + Box-Muller

import math
import time
from statistics import NormalDist

import numpy as np

_LOI_NORMALE = NormalDist()


# generateur congruentiel x_(n+1) = (a x_n + c) mod 2^64
# hull-dobell : c impair, a - 1 multiple de 4, donc periode 2^64
class GenerateurCongruentiel:
    A = 6364136223846793005
    C = 1442695040888963407
    M = 2**64

    def __init__(self, graine=2026):
        self.etat = int(graine) % self.M

    # n entiers suivants, par doublement : x_(i+k) = a_k x_i + c_k
    # avec a_2k = a_k^2 et c_2k = c_k (a_k + 1), meme suite que la boucle naive
    def entiers(self, n):
        if n <= 0:
            return np.empty(0, dtype=np.uint64)
        masque = self.M - 1
        x = np.empty(n, dtype=np.uint64)
        x[0] = (self.A * self.etat + self.C) & masque
        k, a_k, c_k = 1, self.A, self.C
        while k < n:
            j = min(k, n - k)
            x[k:k + j] = np.uint64(a_k) * x[:j] + np.uint64(c_k)
            c_k = (c_k * (a_k + 1)) & masque
            a_k = (a_k * a_k) & masque
            k += j
        self.etat = int(x[-1])
        return x

    # uniformes sur ]0, 1[ : 52 bits de poids fort
    def uniformes(self, n):
        x = self.entiers(n)
        return ((x >> np.uint64(12)).astype(np.float64) + 0.5) * 2.0**-52

    # box-muller
    def normales(self, n):
        n_paires = (n + 1) // 2
        u = self.uniformes(2 * n_paires)
        rayon = np.sqrt(-2.0 * np.log(u[0::2]))
        angle = 2.0 * np.pi * u[1::2]
        z = np.concatenate((rayon * np.cos(angle), rayon * np.sin(angle)))
        return z[:n]


# trajectoires, schema exact ; ligne j = cours a la date t_j
def simuler_trajectoires(spot, volatilite, taux, dividende, maturite,
                         n_pas, n_simulations, generateur):
    dt = maturite / n_pas
    derive = (taux - dividende - 0.5 * volatilite**2) * dt
    diffusion = volatilite * math.sqrt(dt)
    S = np.empty((n_pas + 1, n_simulations))
    S[0] = spot
    for j in range(1, n_pas + 1):
        z = generateur.normales(n_simulations)
        S[j] = S[j - 1] * np.exp(derive + diffusion * z)
    return S


def _payoff(cours, strike, type_option):
    if type_option == "put":
        return np.maximum(strike - cours, 0.0)
    return np.maximum(cours - strike, 0.0)


def _verifier_type(type_option):
    type_option = str(type_option).lower()
    if type_option not in ("call", "put"):
        raise ValueError("type_option doit valoir 'call' ou 'put'")
    return type_option


# pricer americain LSM ; n_pas = nombre total de dates d'exercice
def prix_americaine_lsm(spot, strike, volatilite, taux, maturite, dividende=0.0,
                        type_option="put", degre=3, itm_seulement=True,
                        n_simulations=100_000, n_pas=50, graine=2026,
                        niveau_confiance=0.95):
    debut = time.perf_counter()
    type_option = _verifier_type(type_option)
    if degre < 0 or int(degre) != degre:
        raise ValueError("degre doit etre un entier positif")
    degre = int(degre)

    generateur = GenerateurCongruentiel(graine)
    S = simuler_trajectoires(spot, volatilite, taux, dividende, maturite,
                             n_pas, n_simulations, generateur)
    actu = math.exp(-taux * maturite / n_pas)

    # flux recu par chaque trajectoire, actualise a la date courante
    flux = _payoff(S[-1], strike, type_option)
    flux_europeen = math.exp(-taux * maturite) * flux

    for j in range(n_pas - 1, 0, -1):
        flux *= actu
        exercice = _payoff(S[j], strike, type_option)

        if itm_seulement:
            idx = np.flatnonzero(exercice > 0.0)
        else:
            idx = np.arange(n_simulations)
        if idx.size <= degre:
            continue

        X = np.vander(S[j, idx] / strike, degre + 1, increasing=True)
        coefficients = np.linalg.lstsq(X, flux[idx], rcond=None)[0]
        continuation = X @ coefficients

        exercer = idx[(exercice[idx] > continuation) & (exercice[idx] > 0.0)]
        flux[exercer] = exercice[exercer]

    flux *= actu

    prix = float(flux.mean())
    erreur = float(flux.std(ddof=1) / math.sqrt(n_simulations))
    exercice_immediat = float(_payoff(np.float64(spot), strike, type_option))
    if exercice_immediat > prix:
        prix, erreur = exercice_immediat, 0.0

    quantile = _LOI_NORMALE.inv_cdf(0.5 + niveau_confiance / 2.0)
    demi_largeur = quantile * erreur
    prix_europeen = float(flux_europeen.mean())
    return {
        "prix": prix,
        "erreur_standard": erreur,
        "demi_largeur_ic": demi_largeur,
        "ic_bas": prix - demi_largeur,
        "ic_haut": prix + demi_largeur,
        "prix_europeen_mc": prix_europeen,
        "erreur_standard_europeen": float(flux_europeen.std(ddof=1) / math.sqrt(n_simulations)),
        "prime_exercice_anticipe": prix - prix_europeen,
        "n_simulations": n_simulations,
        "temps_calcul": time.perf_counter() - debut,
    }


# black-scholes-merton
def prix_europeenne_bs(spot, strike, volatilite, taux, maturite, dividende=0.0,
                       type_option="put"):
    type_option = _verifier_type(type_option)
    d1 = (math.log(spot / strike) + (taux - dividende + 0.5 * volatilite**2) * maturite) \
        / (volatilite * math.sqrt(maturite))
    d2 = d1 - volatilite * math.sqrt(maturite)
    N = _LOI_NORMALE.cdf
    spot_actualise = spot * math.exp(-dividende * maturite)
    strike_actualise = strike * math.exp(-taux * maturite)
    if type_option == "call":
        return spot_actualise * N(d1) - strike_actualise * N(d2)
    return strike_actualise * N(-d2) - spot_actualise * N(-d1)


# arbre crr ; n_dates_exercice -> option bermudienne (n_pas doit en etre un multiple)
def prix_arbre_binomial(spot, strike, volatilite, taux, maturite, dividende=0.0,
                        type_option="put", americaine=True, n_pas=5000,
                        n_dates_exercice=None):
    type_option = _verifier_type(type_option)
    if n_dates_exercice is None:
        ecart = 1
    else:
        if n_pas % n_dates_exercice != 0:
            raise ValueError("n_pas doit etre un multiple de n_dates_exercice")
        ecart = n_pas // n_dates_exercice

    dt = maturite / n_pas
    u = math.exp(volatilite * math.sqrt(dt))
    p = (math.exp((taux - dividende) * dt) - 1.0 / u) / (u - 1.0 / u)
    actu = math.exp(-taux * dt)

    # noeud k de l'etape i : S = spot * u^(2k - i)
    valeurs = _payoff(spot * u ** (2.0 * np.arange(n_pas + 1) - n_pas), strike, type_option)
    for i in range(n_pas - 1, -1, -1):
        valeurs = actu * (p * valeurs[1:] + (1.0 - p) * valeurs[:-1])
        if americaine and i % ecart == 0:
            cours = spot * u ** (2.0 * np.arange(i + 1) - i)
            valeurs = np.maximum(valeurs, _payoff(cours, strike, type_option))
    return float(valeurs[0])
