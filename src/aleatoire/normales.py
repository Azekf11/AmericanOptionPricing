from math import e, pi, sqrt

import numpy as np

from src.aleatoire.lcg import Uniformes


METHODES = ("box-muller", "inverse", "acceptation-rejet")

# constante d'acceptation-rejet (demi-normale / exponentielle)
C_AR = sqrt(2.0 * e / pi)


# coefficients ppnd16 (wichura, AS 241)
A = (3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
     1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
     3.3430575583588128105e4, 2.5090809287301226727e3)
B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2, 5.3941960214247511077e3,
     2.1213794301586595867e4, 3.9307895800092710610e4, 2.8729085735721942674e4,
     5.2264952788528545610e3)
C = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
     3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
     2.27238449892691845833e-2, 7.74545014278341407640e-4)
D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
     1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
     1.05075007164441684324e-9)
E = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
     2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
     2.71155556874348757815e-5, 2.01033439929228813265e-7)
F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
     7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
     2.04426310338993978564e-15)


# polynome (horner)
def horner(coefs, x):
    res = np.zeros_like(x)
    for c in reversed(coefs):
        res = res * x + c
    return res


# quantile de N(0,1)
def ppnd16(p):
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    z = np.empty_like(p)

    # zone centrale
    centre = np.abs(q) <= 0.425
    r = 0.180625 - q[centre] ** 2
    z[centre] = q[centre] * horner(A, r) / horner(B, r)

    # queues
    queue = ~centre
    r = np.where(q[queue] < 0, p[queue], 1.0 - p[queue])
    r = np.sqrt(-np.log(r))
    proche = r <= 5.0

    v = np.empty_like(r)
    v[proche] = horner(C, r[proche] - 1.6) / horner(D, r[proche] - 1.6)
    v[~proche] = horner(E, r[~proche] - 5.0) / horner(F, r[~proche] - 5.0)
    z[queue] = np.where(q[queue] < 0, -v, v)

    return z


class Normales:
    # tirages N(0,1) a partir de nos uniformes (jamais numpy.random)

    def __init__(self, methode="box-muller", generateur="park-miller", graine=42):
        if methode not in METHODES:
            raise ValueError("methode inconnue : " + str(methode))
        self.methode = methode
        self.u = Uniformes(generateur, graine)

    def tirer(self, n):
        if n <= 0:
            return np.empty(0)
        if self.methode == "box-muller":
            return self.box_muller(n)
        if self.methode == "inverse":
            return ppnd16(self.u.tirer(n))
        return self.acceptation_rejet(n)

    # box-muller
    def box_muller(self, n):
        n_paires = (n + 1) // 2
        u = self.u.tirer(2 * n_paires)
        u1, u2 = u[0::2], u[1::2]

        # r = -2 ln(u1), v = 2 pi u2
        rayon = np.sqrt(-2.0 * np.log(u1))
        angle = 2.0 * pi * u2

        z = np.empty(2 * n_paires)
        z[0::2] = rayon * np.cos(angle)
        z[1::2] = rayon * np.sin(angle)
        return z[:n]

    # acceptation-rejet
    def acceptation_rejet(self, n):
        res = np.empty(n)
        rempli = 0

        while rempli < n:
            manque = n - rempli
            nb = int(manque * C_AR * 1.05) + 16
            u = self.u.tirer(3 * nb).reshape(nb, 3)

            x = -np.log(u[:, 0])                            # x ~ exp(1)
            ok = u[:, 1] <= np.exp(-0.5 * (x - 1.0) ** 2)   # test
            signe = np.where(u[:, 2] < 0.5, -1.0, 1.0)      # signe

            v = (signe * x)[ok][:manque]
            res[rempli:rempli + v.size] = v
            rempli += v.size

        return res
