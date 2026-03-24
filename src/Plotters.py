import matplotlib.pyplot as plt
import math

def plot_frontier(res, L):
    volatility = [math.sqrt(v) for (v, _, _) in res]
    returns = [ret for (_, ret, _) in res]
    w_s = [w for (_,_,w) in res]

    plt.plot(volatility, returns, label=f"L = {L}")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()