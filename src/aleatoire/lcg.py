import numpy as np


# park-miller (tableau du cours)
A_PM = 16807
M_PM = 2**31 - 1

# wichmann-hill (annexe du poly)
A_WH = (171, 172, 170)
M_WH = (30269, 30307, 30323)

TAILLE_BLOC = 2**14


class LCGBlocs:
    # lcg x <- a*x mod m
    # calcule par blocs : x(n+B) = a^B * x(n) mod m, meme suite que la boucle classique

    def __init__(self, a, m, x0, taille_bloc=TAILLE_BLOC):
        if x0 % m == 0:
            raise ValueError("x0 = 0 interdit (la suite reste bloquee a 0)")

        # premier bloc : boucle classique
        x = x0 % m
        bloc = np.empty(taille_bloc, dtype=np.int64)
        for i in range(taille_bloc):
            x = (a * x) % m
            bloc[i] = x

        self.m = m
        self.saut = pow(a, taille_bloc, m)
        self.bloc = bloc
        self.pos = 0

    def entiers(self, n):
        sortie = np.empty(n, dtype=np.int64)
        rempli = 0

        while rempli < n:
            # bloc suivant
            if self.pos == self.bloc.size:
                self.bloc = (self.bloc * self.saut) % self.m
                self.pos = 0

            k = min(self.bloc.size - self.pos, n - rempli)
            sortie[rempli:rempli + k] = self.bloc[self.pos:self.pos + k]
            self.pos += k
            rempli += k

        return sortie


class Uniformes:
    # uniformes sur ]0, 1[
    # graine = numero du bloc de la suite (blocs disjoints, pas x0 = graine)

    BLOC_PM = 2**25
    GRAINE_MAX_PM = 63
    BLOC_WH = 2**36
    GRAINE_MAX_WH = 100

    def __init__(self, nom="park-miller", graine=42):
        graine = int(graine)
        self.nom = nom
        self.utilisees = 0

        if nom == "park-miller":
            if not 1 <= graine <= self.GRAINE_MAX_PM:
                raise ValueError("graine park-miller entre 1 et 63")
            self.capacite = self.BLOC_PM
            x0 = pow(A_PM, graine * self.BLOC_PM, M_PM)
            self.lcg = [LCGBlocs(A_PM, M_PM, x0)]

        elif nom == "wichmann-hill":
            if not 1 <= graine <= self.GRAINE_MAX_WH:
                raise ValueError("graine wichmann-hill entre 1 et 100")
            self.capacite = self.BLOC_WH
            depart = (5, 10, 15)
            etats = [(pow(a, graine * self.BLOC_WH, m) * s) % m
                     for a, m, s in zip(A_WH, M_WH, depart)]
            self.lcg = [LCGBlocs(a, m, x) for a, m, x in zip(A_WH, M_WH, etats)]

        else:
            raise ValueError("generateur inconnu : " + str(nom))

    def tirer(self, n):
        # on ne deborde pas sur le bloc de la graine suivante
        self.utilisees += n
        if self.utilisees > self.capacite:
            raise RuntimeError("bloc de la graine epuise : reduire N ou passer a wichmann-hill")

        # park-miller
        if self.nom == "park-miller":
            return self.lcg[0].entiers(n) / M_PM

        # wichmann-hill
        x, y, z = (g.entiers(n) for g in self.lcg)
        u = x / M_WH[0] + y / M_WH[1] + z / M_WH[2]
        u -= np.floor(u)
        u[u == 0.0] = 0.5 / M_WH[0]
        return u
