import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open("summaryPlots.json", "w")
    to_do = json.load(to_do, f)
    f.close()

    n_experiments = len(to_do)

    data = []

    for id_experiment in range(n_experiments):
        data.append(to_do[id_experiment]["metric_tests"])

    plt.boxplot(data)
    plt.show()



