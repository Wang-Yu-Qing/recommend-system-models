from models.Popular import Popular
from models.Random import Random
from models.ItemCF import ItemCF
from models.UserCF import UserCF
from models.TagBasic import TagBasic
from models.LFM import LFM
from utils.Data_util import Data_util
import pandas as pd


def run_model(model_type, data_type, **kwargs):
    DU = Data_util(data_type)
    if model_type == "UserCF":
        train_data, test_data = DU.read_event_data()
        model = UserCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'],
                       timestamp=kwargs['timestamp'])
    elif model_type == "ItemCF":
        train_data, test_data = DU.read_event_data()
        model = ItemCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'],
                       timestamp=kwargs['timestamp'])
    elif model_type == "LFM":
        samples = DU.build_samples(data_type, neg_frac=kwargs['neg_frac'])  # noqa
        train_data, test_data = DU.split_samples(0.25, samples)
        model = LFM(data_type=data_type, n=kwargs['n'],
                    neg_frac_in_train=kwargs['neg_frac'],
                    hidden_dim=kwargs['dim'])
    elif model_type == "Random":
        train_data, test_data = DU.read_event_data()
        model = Random(data_type=data_type, n=kwargs['n'])
    elif model_type == "MostPopular":
        train_data, test_data = DU.read_event_data()
        model = Popular(data_type=data_type, n=kwargs['n'])
    elif model_type == "TagBasic":
        item_tag, train_data, test_data = DU.read_event_data()
        model = TagBasic(data_type=data_type, n=kwargs['n'])
    else:
        raise ValueError("Invalid model type: {}".format(model_type))
    model.fit(train_data, force_training=kwargs['force_training'])
    model.evaluate(test_data)


if __name__ == '__main__':
    # run_model("TagBasic", "CiteULike", n=20)
    run_model("Random", "MovieLens_100K",
              n=20, force_training=False, timestamp=True)
    run_model("MostPopular", "MovieLens_100K",
              n=20, force_training=False, timestamp=True)
    run_model("UserCF", "MovieLens_100K",
              n=20, k=80, force_training=True, timestamp=True)
    run_model("UserCF", "MovieLens_100K",
              n=20, k=80, force_training=True, timestamp=False)
    run_model("ItemCF", "MovieLens_100K",
              n=20, k=20, force_training=False, timestamp=False)
    run_model("ItemCF", "MovieLens_100K",
              n=20, k=20, force_training=False, timestamp=True)
    run_model("LFM", "MovieLens_100K", n=100, dim=10, neg_frac=20, force_training=True)  # noqa
