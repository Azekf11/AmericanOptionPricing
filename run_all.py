# relance tous les calculs puis les figures (environ 20 minutes)
# python run_all.py            -> tout
# python run_all.py figures    -> seulement les figures (a partir des csv)

import sys
import time

from questions import complements, encadrement, figures, q1_degre, q2_europeenne, q3_convergence


if __name__ == "__main__":
    debut = time.perf_counter()

    if "figures" not in sys.argv:
        q1_degre.main()
        q2_europeenne.main()
        q3_convergence.main()
        complements.main()
        encadrement.main()

    figures.main()
    print(f"termine en {time.perf_counter() - debut:.0f} s")
