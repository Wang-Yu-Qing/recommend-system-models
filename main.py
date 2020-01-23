# from matrix_factorization.model import Matrix_factorization
from userCF.model import UserCF, train_user_cf_model, evaluate_user_cf_model
import pandas as pd
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


if __name__ == '__main__':
    # train_and_evaluate_matrix_factorization_model()
    train_user_cf_model(1, 'data/MovieLens/ratings.csv',
                        'userCF/MovieLens', IIF=True)
    for k, n in [(10, 20), (30, 20), (40, 20), (80, 20), (120, 20), (160, 20)]:
        evaluate_user_cf_model(1, 'data/MovieLens/ratings.csv',
                               'userCF/MovieLens', k, n,
                               ensure_new=True,
                               IIF=True)

    # train_user_cf_model(0.6, 'data/Retailrocket/events.csv',
    #                     'userCF/Retailrocket', IIF=True)
    # for k, n in [(10, 20), (30, 20), (40, 20), (80, 20), (120, 20), (160, 20)]:
    # for k, n in [(280, 20), (380, 20), (500, 20), (700, 20), (800, 20), (900, 20)]:
    #     evaluate_user_cf_model(0.6, 'data/Retailrocket/events.csv',
    #                            'userCF/Retailrocket', k, n,
    #                            ensure_new=True,
    #                            IIF=False)
