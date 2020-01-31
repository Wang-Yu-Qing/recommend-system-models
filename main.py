# from matrix_factorization.model import Matrix_factorization
from userCF.model import UserCF
from itemCF.model import ItemCF
# from LFM.model import train_LFM_model, evaluate_LFM_model
from utils.Data_util import Data_util
import pandas as pd
from preprocess_retailrocket import preprocess_event_data


def prepare_data(data_type):
    train, test = Data_util(data_type).read_event_data()
    # train, test = Data_util('MovieLens_1M').read_event_data()
    del train["timestamp"], test["timestamp"]
    return train, test


def run_model(model_type, data_type, **kwargs):
    train_data, test_data = prepare_data(data_type)
    if model_type == "UserCF":
        model = UserCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'])
    elif model_type == "ItemCF":
        model = ItemCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'])
    elif model_type == "LFM":
        model = LFM(data_type=data_type)
    else:
        raise ValueError("Invalid model type: {}".format(model_type))
    model.fit(train_data)
    model.evaluate(test_data)


if __name__ == '__main__':
    run_model("UserCF", "MovieLens_100K", n=20, k=80)
    run_model("ItemCF", "MovieLens_100K", n=20, k=20)
