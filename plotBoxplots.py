import json
import numpy as np
import matplotlib.pyplot as plt


def read_everything(filesList):
    to_do = None
    for fname in filesList:
        f = open(fname, "r")
        data = json.load(f)
        f.close()
        if to_do is None:
            to_do = data
        else:
            for i in range(len(data)):
                to_do[i]["metric_tests"].extend(data[i]["metric_tests"])
    return to_do



if __name__ == "__main__":
    to_do = read_everything(["summary_plots_2_runs.json", "summary_plots_18_runs.json"])
    n_experiments = len(to_do)
    data = []
    labels = ["a","b","c","d","e","f"]
    
    

    to_do = [to_do[i] for i in [1,3,0,2,5,4]]

    for id_experiment in range(n_experiments):
        print(to_do[id_experiment]["classifier_name"], to_do[id_experiment]["feature"])
        data.append(to_do[id_experiment]["metric_tests"])

    plt.boxplot(data, labels = labels)
    plt.savefig("boxplot.pdf",format="pdf")
    plt.show()



