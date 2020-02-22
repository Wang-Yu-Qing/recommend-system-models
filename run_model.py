from models.Popular import Popular
from models.Random import Random
from models.ItemCF import ItemCF
from models.UserCF import UserCF
from models.TagBasic import TagBasic
from models.LFM import LFM
from models.Wide_and_deep import Wide_and_deep
from utils.Data_util import Data_util
from utils.Feature_util import Feature_util
import pandas as pd


def run(model_type, data_type, **kwargs):
    DU = Data_util(data_type)
    train_data, test_data = DU.read_event_data()
    if model_type == "UserCF":
        model = UserCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'],
                       timestamp=kwargs['timestamp'])
        model.fit(train_data)
    elif model_type == "ItemCF":
        model = ItemCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'],
                       timestamp=kwargs['timestamp'])
        model.fit(train_data)
    elif model_type == "LFM":
        train_data = DU.build_samples(kwargs['neg_frac'], train_data)
        model = LFM(data_type=data_type, n=kwargs['n'],
                    neg_frac_in_train=kwargs['neg_frac'])
        model.fit(train_data)
    elif model_type == "Random":
        model = Random(data_type=data_type, n=kwargs['n'])
        model.fit(train_data)
    elif model_type == "MostPopular":
        model = Popular(data_type=data_type, n=kwargs['n'])
        model.fit(train_data)
    elif model_type == "TagBasic":
        model = TagBasic(data_type=data_type, n=kwargs['n'], k=kwargs['k'])
        model.fit(train_data)
    elif model_type == "Wide&Deep":
        # create negative samples (only for training set)
        train_data = DU.build_samples(kwargs['neg_frac'], train_data)
        # get user, item features
        FU = Feature_util(data_type)
        users_info, items_info = FU.read_user_item_info()
        del items_info["title"], items_info["video_release_date"], items_info["URL"]
        # join
        assert train_data["visitorid"].dtype == users_info["visitorid"].dtype
        assert train_data["itemid"].dtype == items_info["itemid"].dtype
        train_data = DU.join_movie_lens_event_data(train_data, users_info, items_info)
        model = Wide_and_deep(data_type=data_type, neg_frac_in_train=kwargs["neg_frac"], n=kwargs["n"])
        model.fit(train_data, users_info, items_info)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))
    # make sure test data contains only event data
    evaluation_result = model.evaluate(test_data)
    return evaluation_result


if __name__ == '__main__':
    run("Wide&Deep", "MovieLens_100K", n=20, neg_frac=40, test_size=0.25)
    # run("LFM", "MovieLens_100K", n=100, dim=10, neg_frac=40, test_size=0.25)
    # run("TagBasic", "Hetrec-2k", n=20, k=2)
    # run("Random", "Hetrec-2k", n=20, timestamp=True)
    # run("MostPopular", "Hetrec-2k", n=20, timestamp=True)
    # run("Random", "MovieLens_100K", n=20, timestamp=True)
    # run("MostPopular", "MovieLens_100K", n=20, timestamp=True)
    # run("UserCF", "MovieLens_100K", n=20, k=80, timestamp=True)
    # run("ItemCF", "MovieLens_100K", n=20, k=20, timestamp=True)
