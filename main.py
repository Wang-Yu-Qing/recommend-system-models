# from matrix_factorization.model import Matrix_factorization
from userCF.model import UserCF
import pandas as pd
import pickle
from preprocess_retailrocket import preprocess_event_data


def train_and_evaluate_matrix_factorization_model():
    events = pd.read_csv('data/Retailrocket/events.csv')
    n_visitors = len(pd.unique(events['visitorid']))
    n_items = len(pd.unique(events['itemid']))
    print('n visitors: {}, n items: {}'.format(n_visitors, n_items))
    train, test = preprocess_event_data(events)
    m = Matrix_factorization({'item_embed_input_dim': n_items,
                              'item_embed_output_dim': 5,
                              'visitor_embed_input_dim': n_visitors,
                              'visitor_embed_output_dim': 5})
    m.construct()
    m.train(train)


def train_user_fc_model():
    event_data = pd.read_csv('data/Retailrocket/events.csv')
    # use small set for limited memory
    # event_data = event_data.sample(frac=0.6)
    event_data = event_data.iloc[:int(len(event_data)*0.6), :]
    split = int(len(event_data)*0.8)
    train = event_data.iloc[:split, :]
    test = event_data.iloc[split:, :]
    users_id = pd.unique(train['visitorid'])
    items_id = pd.unique(train['itemid'])
    model = UserCF(users_id, 3)
    model.form_users_distance_matrix(train, metric='cosine')
    model.save('userCF')


if __name__ == '__main__':
    # train_and_evaluate_matrix_factorization_model()
    train_user_fc_model()
    raise SystemExit
    model = UserCF([], 200)
    model.load('userCF')
    event_data = pd.read_csv('data/Retailrocket/events.csv')
    event_data = event_data.iloc[:int(len(event_data)*0.6), :]
    bool_index = event_data['event'] != 1
    event_data = event_data.loc[bool_index, :]
    split = int(len(event_data)*0.8)
    test = event_data.iloc[split:, :]
    model.evaluate(test, 'prediction')
