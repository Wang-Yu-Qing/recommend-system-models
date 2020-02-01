# from matrix_factorization.model import Matrix_factorization
from userCF.model import UserCF
from itemCF.model import ItemCF
from LFM.model import LFM
from utils.Data_util import Data_util
import pandas as pd


def prepare_data(data_type):
    train, test = Data_util(data_type).read_event_data()
    del train["timestamp"], test["timestamp"]
    return train, test


def create_negative_samples(data_type, neg_frac):
    """ return all samples
    """
    data_util = Data_util(data_type)
    pos_samples = data_util.read_event_data(test_size=0)[0]
    del pos_samples['rating'], pos_samples['timestamp']
    pos_samples['event'] = 1
    neg_samples = data_util.create_negative_samples(pos_samples, neg_frac)
    samples = pd.concat([neg_samples, pos_samples], ignore_index=True)
    samples = samples.sample(frac=1, random_state=100).reset_index(drop=True)
    return samples


def split_samples(test_size, samples):
    split = int(len(samples)*test_size)
    test, train = samples.iloc[:split, :], samples.iloc[split:, :]
    return train, test


def run_model(model_type, data_type, **kwargs):
    if model_type == "UserCF":
        train_data, test_data = prepare_data(data_type)
        model = UserCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'])
    elif model_type == "ItemCF":
        train_data, test_data = prepare_data(data_type)
        model = ItemCF(data_type=data_type, n=kwargs['n'], k=kwargs['k'])
    elif model_type == "LFM":
        samples = create_negative_samples(data_type, neg_frac=kwargs['neg_frac'])  # noqa
        train_data, test_data = split_samples(0.25, samples)
        model = LFM(data_type=data_type, n=kwargs['n'],
                    neg_frac_in_train=kwargs['neg_frac'],
                    hidden_dim=kwargs['dim'])
    else:
        raise ValueError("Invalid model type: {}".format(model_type))
    model.fit(train_data, force_training=kwargs['force_training'])
    model.evaluate(test_data)


if __name__ == '__main__':
    # run_model("UserCF", "MovieLens_100K", n=20, k=80)
    # run_model("ItemCF", "MovieLens_100K", n=20, k=20)
    run_model("LFM", "MovieLens_100K", n=100, dim=10, neg_frac=20, force_training=True)  # noqa
