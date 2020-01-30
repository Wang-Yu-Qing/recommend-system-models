# from matrix_factorization.model import Matrix_factorization
from userCF.model import UserCF
from itemCF.model import train_item_cf_model, evaluate_item_cf_model
from LFM.model import train_LFM_model, evaluate_LFM_model
import pandas as pd
from preprocess_retailrocket import preprocess_event_data

def prepare_data():
    df = pd.read_csv('data/MovieLens/ratrings.csv')
    return df



if __name__ == '__main__':
    event_data = prepare_data()
    user_cf_model = UserCF(20, 'userCF')
    user_cf_model.fit(event_data)
