from utils.Evaluation import Evaluation

if __name__ == "__main__":
    for model_type in ["UserCF", "ItemCF", "LFM", "Wide&Deep", "MostPopular", "Random"]:
        EU = Evaluation(model_type, "MovieLens_100K")
        # EU.compute_recall_precision_pairs()
        EU.compute_AUC() 