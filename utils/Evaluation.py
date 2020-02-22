import numpy as np
import pandas as pd
from run_model import run
"https://docs.scipy.org/doc/numpy/reference/generated/numpy.trapz.html"


class Evaluation:
    def __init__(self, model_type, data_type):
        self.model_type = model_type
        self.data_type = data_type
        self.min_n = 1
        self.max_n = 100
        if model_type in ["UserCF", "ItemCF", "TagBasic"]:
            self.model_kwargs = {"k": 80, "timestamp": True}
        elif model_type in ["LFM", "Wide&Deep"]:
            self.model_kwargs = {"neg_frac": 40, "test_size": 0.25}
        else:
            self.model_kwargs = {}
    
    def compute_recall_precision_pairs(self):
        pairs = []
        for n_reco in range(self.min_n, self.max_n+1, 5):
            self.model_kwargs["n"] = n_reco
            result = run(self.model_type, self.data_type, **self.model_kwargs)
            print("*"*10, " ", n_reco, " ", "*"*10)
            print(result)
            pairs.append((n_reco, result["recall"], result["fallout"]))
        df = pd.DataFrame(pairs, columns=["n", "recall", "fallout"])
        df.to_csv("evaluation_results/{}.csv".format(self.model_type), index=False)
    
    def compute_AUC(self):
        df = pd.read_csv("evaluation_results/{}.csv".format(self.model_type))
        AUC = np.trapz(y=df["recall"], x=df["fallout"])
        print(self.model_type, " AUC: ", AUC)