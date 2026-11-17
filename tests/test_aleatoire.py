from math import sqrt

import numpy as np
import pytest

from src.aleatoire.lcg import A_PM, A_WH, M_PM, M_WH, Uniformes


N_TEST = 400_000


def test_park_miller_meme_suite_que_la_boucle():
    x = pow(A_PM, 42 * Uniformes.BLOC_PM, M_PM)
    attendu = []
    for _ in range(50_000):
        x = (A_PM * x) % M_PM
        attendu.append(x / M_PM)
    assert np.array_equal(np.array(attendu), Uniformes("park-miller", 42).tirer(50_000))


def test_wichmann_hill_meme_suite_que_la_boucle():
    L = Uniformes.BLOC_WH
    xyz = [(pow(a, 7 * L, m) * s) % m for a, m, s in zip(A_WH, M_WH, (5, 10, 15))]
    attendu = []
    for _ in range(40_000):
        xyz = [(a * v) % m for a, v, m in zip(A_WH, xyz, M_WH)]
        s = sum(v / m for v, m in zip(xyz, M_WH))
        attendu.append(s - int(s))
    assert np.allclose(np.array(attendu), Uniformes("wichmann-hill", 7).tirer(40_000), rtol=0, atol=1e-15)


@pytest.mark.parametrize("graine", [0, -3, 64])
def test_graine_refusee(graine):
    with pytest.raises(ValueError):
        Uniformes("park-miller", graine)


def test_bloc_epuise():
    u = Uniformes("park-miller", 1)
    u.capacite = 1000
    u.tirer(1000)
    with pytest.raises(RuntimeError):
        u.tirer(1)


@pytest.mark.parametrize("nom", ["park-miller", "wichmann-hill"])
def test_uniformes_moments(nom):
    u = Uniformes(nom, 17).tirer(N_TEST)
    assert u.min() > 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 4 * sqrt(1 / 12 / N_TEST)
    assert abs(u.var() - 1 / 12) < 0.002


@pytest.mark.parametrize("nom", ["park-miller", "wichmann-hill"])
def test_graines_independantes(nom):
    # avec x0 = 1 puis x0 = 2 on aurait u2 = 2 u1 mod 1 (correlation 0,5)
    u = [Uniformes(nom, g).tirer(N_TEST) for g in (1, 2, 3)]
    seuil = 4 / sqrt(N_TEST)
    assert abs(np.corrcoef(u[0], u[1])[0, 1]) < seuil
    assert abs(np.corrcoef(u[0], u[2])[0, 1]) < seuil
    assert abs(np.corrcoef(u[0], np.mod(2 * u[1], 1.0))[0, 1]) < seuil


@pytest.mark.parametrize("nom", ["park-miller", "wichmann-hill"])
def test_reproductible(nom):
    a = Uniformes(nom, 12).tirer(1000)
    b = Uniformes(nom, 12).tirer(1000)
    c = Uniformes(nom, 13).tirer(1000)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
