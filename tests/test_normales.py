from math import erf, sqrt

import numpy as np
import pytest

from src.aleatoire.normales import METHODES, Normales, ppnd16


N_TEST = 400_000


def phi(x):
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


# statistique de kolmogorov-smirnov contre N(0,1)
def ks(z):
    x = np.sort(z)
    n = x.size
    F = phi(x)
    return max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))


def test_ppnd16_valeurs():
    p = np.array([0.5, 0.975, 0.995, 0.025, 1e-10])
    attendu = np.array([0.0, 1.959963984540054, 2.5758293035489004, -1.959963984540054, -6.361340902404056])
    assert np.allclose(ppnd16(p), attendu, rtol=0, atol=1e-12)


def test_ppnd16_inverse_de_phi():
    p = np.linspace(1e-6, 1 - 1e-6, 10_001)
    assert np.max(np.abs(phi(ppnd16(p)) - p)) < 1e-14


@pytest.mark.parametrize("generateur", ["park-miller", "wichmann-hill"])
@pytest.mark.parametrize("methode", METHODES)
def test_normales(methode, generateur):
    z = Normales(methode, generateur, 17).tirer(N_TEST)
    assert abs(z.mean()) < 4 / sqrt(N_TEST)
    assert abs(z.var() - 1) < 4 * sqrt(2 / N_TEST)
    assert abs(np.mean(z**3)) < 4 * sqrt(6 / N_TEST)
    # kurtosis : seuil 5 (estimateur a queues epaisses)
    assert abs(np.mean(z**4) - 3) < 5 * sqrt(24 / N_TEST)
    # ks a 1 %
    assert ks(z[:100_000]) < 1.63 / sqrt(100_000)


def test_etat_conserve():
    g = Normales("box-muller", "park-miller", 5)
    a, b = g.tirer(1000), g.tirer(1000)
    assert not np.array_equal(a, b)
    assert np.array_equal(np.concatenate([a, b]), Normales("box-muller", "park-miller", 5).tirer(2000))
