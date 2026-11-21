# American Option Pricing (Longstaff-Schwartz)

Projet Pricing ING4 2026-2027, énoncé 1 (ECE Paris, majeure finance quantitative).

On price des calls et des puts américains par Monte-Carlo. À chaque date d'exercice on compare
le payoff immédiat à la valeur de continuation, estimée par une régression sur les trajectoires
(méthode de Longstaff et Schwartz, 2001).


## Lancer le code

Python 3.10 ou plus.

```
pip install -r requirements.txt

python main.py              # un prix, les paramètres sont en haut du fichier
python -m pytest            # les tests (une dizaine de secondes)
python run_all.py           # recalcule tous les résultats (environ 20 min)
python run_all.py figures   # refait seulement les figures à partir des CSV
```

Une question seule se lance depuis la racine, par exemple `python -m questions.q3_convergence`.

Depuis Python :

```python
from src.pricer.longstaff_schwartz import prix_option_americaine

res = prix_option_americaine("put", S0=36, K=40, T=1, r=0.06, sigma=0.20, q=0.0,
                             degre=3, itm_seulement=True, n_simulations=100_000)
print(res["prix"], res["erreur_type"], res["ic95"], res["temps"])
```

| paramètre | rôle | défaut |
|---|---|---|
| `type_option` | `"call"` ou `"put"` | |
| `S0, K, T, r, sigma, q` | spot, strike, maturité, taux, volatilité, dividende continu | `q = 0` |
| `degre` | degré du polynôme de régression | 3 |
| `itm_seulement` | régression sur les trajectoires dans la monnaie (`True`) ou sur toutes | `True` |
| `n_simulations` | nombre de trajectoires | 100 000 |
| `n_dates` | dates d'exercice par an | 50 |
| `methode_normale` | `"box-muller"`, `"inverse"` ou `"acceptation-rejet"` | `"box-muller"` |
| `generateur_uniforme` | `"park-miller"` ou `"wichmann-hill"` | `"park-miller"` |
| `graine` | numéro de bloc de la suite (1 à 63 pour Park-Miller) | 42 |
| `antithetique` | variables antithétiques | `False` |
| `controle_europeen` | variable de contrôle européenne | `False` |
| `plancher_europeen` | refuse un exercice qui rapporte moins que l'européenne | `False` |
| `hors_echantillon` | règle estimée sur un jeu de trajectoires, appliquée sur un autre | `False` |
| `base_regression` | `"monomes"` ou `"laguerre"` | `"monomes"` |

Avec les valeurs par défaut, c'est l'algorithme du sujet sans rien en plus.


## Organisation

```
AmericanOptionPricing/
    main.py             exemple
    run_all.py          relance tous les calculs
    src/
        aleatoire/      générateurs uniformes (Park-Miller, Wichmann-Hill) et loi normale
        modele/         payoff, Black-Scholes, arbre CRR, trajectoires
        pricer/         longstaff_schwartz.py (la fonction du sujet), bornes.py
        calibration/    TotalEnergies
    questions/          un script par question + vérifications + figures
    tests/              tests pytest
    validation/         second pricer, écrit séparément, pour comparer
    donnees/            cours et dividendes TotalEnergies
    resultats/          tableaux CSV et figures
```


## Nos choix

- Python plutôt que VBA, pour pouvoir vectoriser. numpy ne sert jamais à tirer des nombres
  aléatoires.
- Uniformes : LCG de Park et Miller (a = 16 807, m = 2^31 − 1), le premier du tableau du cours.
  On le calcule par blocs, ce qui donne exactement la même suite qu'une boucle (testé).
- Normales : Box-Muller par défaut parce que c'est le plus rapide. L'inverse (PPND16) et
  l'acceptation-rejet sont codés aussi.
- Graines : avec x0 = 1 puis x0 = 2, les deux suites sont corrélées à 0,5. Une graine est donc un
  numéro de bloc de 2^25 valeurs, et deux graines ne partagent aucune valeur.
- Trajectoires : schéma exact du mouvement brownien géométrique.
- 50 dates d'exercice par an, comme dans l'article. La référence exacte est donc un arbre CRR
  bermudien aux mêmes dates (4,478 sur le cas de l'article, la valeur publiée).
- Régression sur 1, x, ..., x^d avec x = S/K. Degré 3 et trajectoires dans la monnaie, d'après
  la question 1.
- Plancher européen (option) : sur un put TotalEnergies, l'algorithme brut donnait une
  américaine moins chère que l'européenne. Le plancher refuse les exercices qui rapportent moins
  que l'européenne de même échéance.
- Bornes : la borne basse applique la règle estimée à des trajectoires neuves. La borne haute
  vient de la méthode duale, avec des espérances à un pas calculées par Gauss-Hermite.
- TotalEnergies parce que son dividende (4,45 %) est plus haut que le taux (€STR, 2,44 %) : c'est
  le cas où exercer un call avant l'échéance peut valoir le coup. Données Yahoo Finance du
  11/11/2026, déjà dans `donnees/`.
- `validation/` : un second pricer écrit séparément (autre générateur, LCG modulo 2^64). Les deux
  donnent le même prix aux erreurs types près.


## Quelques résultats

Put de l'article (S0 = 36, K = 40, r = 6 %, σ = 20 %, T = 1) :

| | prix |
|---|---|
| européen, Black-Scholes | 3,8443 |
| arbre CRR bermudien, 50 dates | 4,4780 |
| arbre CRR américain | 4,4867 |
| LSM, 100 000 trajectoires, graine 42 | 4,486 ± 0,009 |
| borne basse | 4,4746 ± 0,0008 |
| borne haute duale | 4,4797 ± 0,0002 |

Le reste est dans `resultats/tableaux/`.


## Limites

Biais bas de la règle estimée, option bermudienne et pas vraiment américaine, volatilité
constante, dividende continu au lieu de quatre versements par an, taux plat.


## Références

- F. Longstaff, E. Schwartz, Valuing American options by simulation: a simple least-squares
  approach, Review of Financial Studies, 2001.
- S. Park, K. Miller, Random number generators: good ones are hard to find, Communications of
  the ACM, 1988.
- P. Glasserman, Monte Carlo Methods in Financial Engineering, Springer, 2004.
- Cours de Pricing Monte-Carlo, ECE 2026, et polycopié du cours.
