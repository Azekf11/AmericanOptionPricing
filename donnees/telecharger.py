# telecharge les cours et dividendes de TotalEnergies (TTE.PA)
# les CSV utilises sont deja dans ce dossier (extraction du 11/11/2026)
# pip install yfinance  puis  python donnees/telecharger.py

from pathlib import Path

import yfinance as yf


DOSSIER = Path(__file__).resolve().parent
TICKER = "TTE.PA"
DEBUT, FIN = "2024-11-01", "2026-11-11"


if __name__ == "__main__":
    # cours
    cours = yf.download(TICKER, start=DEBUT, end=FIN, interval="1d", auto_adjust=False, progress=False)
    cours.columns = [c[0] if isinstance(c, tuple) else c for c in cours.columns]
    cours.to_csv(DOSSIER / "TTE_PA_2024-11_2026-11.csv")

    # dividendes
    div = yf.Ticker(TICKER).dividends
    div = div[div.index >= "2024-01-01"]
    div.index = div.index.strftime("%Y-%m-%d")
    div.to_csv(DOSSIER / "TTE_PA_dividendes.csv", header=["Dividende"])

    print(len(cours), "seances,", len(div), "dividendes")
