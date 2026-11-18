# exemple d'utilisation du pricer : modifier les parametres puis lancer  python main.py

from src.modele.arbre_crr import prix_crr
from src.modele.black_scholes import prix_bs
from src.pricer.longstaff_schwartz import prix_option_americaine, resume


# parametres generaux
type_option = "put"
S0 = 36.0
K = 40.0
T = 1.0
r = 0.06
sigma = 0.20
q = 0.0

# parametres de l'option americaine
degre = 3
itm_seulement = True

# parametres numeriques
n_simulations = 100_000
n_dates = 50                 # dates d'exercice par an
methode_normale = "box-muller"
graine = 42


if __name__ == "__main__":
    res = prix_option_americaine(type_option, S0, K, T, r, sigma, q,
                                 n_simulations=n_simulations, n_dates=n_dates,
                                 degre=degre, itm_seulement=itm_seulement,
                                 methode_normale=methode_normale, graine=graine)

    n_total = res["details"]["n_dates"]

    print(f"{type_option} americain  S0={S0}  K={K}  T={T}  r={r}  sigma={sigma}  q={q}")
    print()
    print("LSM          :", resume(res))
    print(f"exercice anticipe sur {res['details']['part_exercee']:.1%} des trajectoires")
    print()
    print(f"europeen BS  : {prix_bs(type_option, S0, K, T, r, sigma, q):.4f}")
    print(f"CRR bermudien: {prix_crr(type_option, S0, K, T, r, sigma, q, n_pas=40 * n_total, exercice='bermudien', n_dates_exercice=n_total):.4f}")
    print(f"CRR americain: {prix_crr(type_option, S0, K, T, r, sigma, q, n_pas=2000):.4f}")
