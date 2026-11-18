import numpy as np


# payoff d'exercice
def payoff(type_option, S, K):
    if type_option == "call":
        return np.maximum(S - K, 0.0)
    if type_option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("type d'option inconnu : " + str(type_option))
