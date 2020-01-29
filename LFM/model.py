import pandas as pd
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, dot, Dense, Concatenate, Add
from keras.models import load_model
from keras import optimizers
from userCF.model import UserCF


class LFM(UserCF):
    def __init__(self, users_id, items_id,
                 hidden_dim, ensure_new, n,
                 negative_sample_fraction):
        self.n_users = len(users_id)
        self.n_items = len(items_id)
        self.max_user_id = max(users_id)
        self.max_item_id = max(items_id)
        self.users, self.items = {}, {}
        self.hidden_dim = hidden_dim
        self.n = n
        self.ensure_new = ensure_new
        self.negative_sample_fraction = negative_sample_fraction

    def dot_structure(self, item_vec, user_vec):
        # dot product of user vec and item vec
        dot_product = dot([item_vec, user_vec],
                          axes=[1, 1], name='Dot_product')
        # add single dense with bias
        out_put = Dense(units=1, input_shape=(1, ),
                        activation='relu',
                        use_bias=True,
                        name='out_put')(dot_product)
        return out_put

    def adding_structure(self, item_vec, user_vec):
        # add user vec and item vec as a new vec
        adding_layer = Add()([item_vec, user_vec])
        # dense layers with bias
        dense_1 = Dense(units=128, input_shape=(self.hidden_dim, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_1')(adding_layer)
        dense_2 = Dense(units=64, input_shape=(128, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_2')(dense_1)
        dense_3 = Dense(units=32, input_shape=(64, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_3')(dense_2)
        dense_4 = Dense(units=16, input_shape=(32, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_4')(dense_3)
        out_put = Dense(units=1, input_shape=(16, ),
                        activation='relu',
                        use_bias=True,
                        name='output')(dense_4)
        return out_put

    def concat_structure(self, item_vec, user_vec):
        # concat user vec and item vec as a new vec
        concat_layer = Concatenate()([item_vec, user_vec])
        # dense layers with bias
        dense_1 = Dense(units=128, input_shape=(self.hidden_dim*2, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_1')(concat_layer)
        dense_2 = Dense(units=64, input_shape=(128, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_2')(dense_1)
        dense_3 = Dense(units=32, input_shape=(64, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_3')(dense_2)
        dense_4 = Dense(units=16, input_shape=(32, ),
                        activation='relu',
                        use_bias=True,
                        name='dense_4')(dense_3)
        out_put = Dense(units=1, input_shape=(16, ),
                        activation='relu',
                        use_bias=True,
                        name='output')(dense_4)
        return out_put

    def construct_model(self, embeds_merge_type):
        """
            input_dim for embedding layer
            should larger than the maximum possible id number
            in the training and testing set
            otherwise, the embedding rows is not enough thus
            lookup cannot be done which will cause
            tensorflow.python.framework.errors_impl.InvalidArgumentError

            using (max_id+1) as input dim will be all good,
            except that may cause some waste of memory
        """
        # struct for item
        item_input = Input(shape=[1], name='Item')
        item_embed = Embedding(input_dim=self.max_item_id+1,
                               output_dim=self.hidden_dim,
                               input_length=1,
                               name='item_embedding')(item_input)
        item_vec = Flatten(name='item_flatten')(item_embed)
        # struct for user
        user_input = Input(shape=[1], name='User')
        user_embed = Embedding(input_dim=self.max_user_id+1,
                               output_dim=self.hidden_dim,
                               input_length=1,
                               name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embed)
        if embeds_merge_type == 'dot':
            out_put = self.dot_structure(item_vec, user_vec)
        elif embeds_merge_type == 'add':
            out_put = self.adding_structure(item_vec, user_vec)
        elif embeds_merge_type == 'concat':
            out_put = self.concat_structure(item_vec, user_vec)
        else:
            raise ValueError('Invalid embeds merge type provided!')
        self.model = keras.Model([item_input, user_input], out_put)
        optimizer = optimizers.Adam(lr=0.0005)
        self.model.compile(optimizer, 'mse')
        keras.utils.plot_model(self.model,
                               to_file='LFM/model_struc/model_{}.png'.format(embeds_merge_type),
                               show_shapes=True, show_layer_names=True)
        self.model.summary()

    def init_item_and_user_objects(self, event_data):
        super().init_item_and_user_objects(event_data)

    def reset_item_and_user_objects(self, positive_and_negative_samples):
        self.users, self.items = {}, {}
        event_data = positive_and_negative_samples.loc[positive_and_negative_samples['event'] == 1]
        super().init_item_and_user_objects(event_data)

    def create_negative_samples_for_single_user(self, user, items_pop,
                                                negative_samples):
        """
            items pop must be sorted in decreasing order
        """
        n_negative_samples_created = 0
        n_positive_samples = len(user.covered_items)
        n_negative_samples_needed = int(self.negative_sample_fraction * n_positive_samples)
        for item_id, pop in items_pop:
            if item_id not in user.covered_items:
                negative_samples.append([user.id, item_id, 0])
                n_negative_samples_created += 1
            if n_negative_samples_created == n_negative_samples_needed:
                break
        else:
            print("""Not enough untouched items for user {} to create {} negative samples,
                     create {} instead.""".format(user.id, n_negative_samples_needed, n_negative_samples_created))  # noqa

    def create_negative_samples(self, event_data):
        """
        create negative samples for each user
        number of negative samples equals to positive samples
        if the item not touched by the user and its popularity is
        high, then mark this item as user's negative sample
        """
        self.init_item_and_user_objects(event_data)
        # sort items by popularity
        item_pop = {}
        for item_id, item in self.items.items():
            item_pop[item_id] = len(item.covered_users)
        # return list for tuples
        items_pop = sorted(item_pop.items(),
                           key=lambda item: item[1],
                           reverse=True)
        negative_samples = []
        for user in self.users.values():
            self.create_negative_samples_for_single_user(user, items_pop,
                                                         negative_samples)
        return pd.DataFrame(negative_samples,
                            columns=['visitorid', 'itemid', 'event'])

    def save_keras_model(self):
        self.model.save('LFM/MovieLens_model.h5')

    def load_keras_model(self):
        self.model = load_model('LFM/MovieLens_model.h5')

    def train(self, train_data):
        self.model.fit([train_data['itemid'],
                        train_data['visitorid']],
                       train_data['event'],
                       epochs=10)
        # self.save_keras_model()

    def evaluate_mse(self, test_data):
        # input order must corresponding to the model input building order
        result = self.model.evaluate(x=[test_data['itemid'],
                                        test_data['visitorid']],
                                     y=test_data['event'])
        print('mse: ', result)

    def make_recommendation(self, user_id):
        try:
            user = self.users[user_id]
        except KeyError:
            print('User {} has not shown in the training set.'.format(user_id))
            return -1
        items_rank = {}
        # compute user's interest of every item
        for item_id in self.items.keys():
            if self.ensure_new and item_id in user.covered_items:
                continue
            score = self.model.predict([[item_id], [user_id]])
            items_rank[item_id] = score
        reco_items = super().get_top_n_items(items_rank)
        # reco_items = [item_id for item_id, score in items_rank.items() if score >= 0.96]
        print(len(reco_items))
        return reco_items

    def evaluate(self, test_data):
        if not hasattr(self, 'model'):
            raise ValueError('model has not been trained or loaded.')
        # self.evaluate_mse(test_data)
        # drop negative samples for recommendation evaluation
        test_data = test_data.loc[test_data['event'] == 1]
        print(super().evaluate_recommendation(test_data))


def train_LFM_model():
    event_data = pd.read_csv('data/MovieLens/ratings.csv')
    model = LFM(users_id=pd.unique(event_data['visitorid']),
                items_id=pd.unique(event_data['itemid']),
                hidden_dim=10,
                ensure_new=True,
                n=100,
                negative_sample_fraction=5)
    model.construct_model('dot')
    # will record user-items and item-users history using all data
    negative_samples = model.create_negative_samples(event_data)
    event_data = event_data.loc[:, ['visitorid', 'itemid', 'event']]
    # change event as 1
    event_data['event'] = 1
    event_data = pd.concat([event_data, negative_samples],
                           ignore_index=True)
    n_positive_samples = sum(event_data['event'] == 1)
    n_negative_samples = sum(event_data['event'] == 0)
    print('positive/negative: {}/{}'.format(n_positive_samples,
                                            n_negative_samples))
    train, test = train_test_split(event_data, test_size=0.25,
                                   random_state=1, shuffle=True)
    train.to_csv('LFM/train_movielens.csv', index=False)
    test.to_csv('LFM/test_movielens.csv', index=False)
    # reset item and user objects, only record training data
    model.reset_item_and_user_objects(train)
    model.train(train)
    model.save('LFM/LFM_model.pickle')


def evaluate_LFM_model():
    event_data = pd.read_csv('data/MovieLens/ratings.csv')
    model = LFM.load('LFM/LFM_model.pickle')
    model.evaluate(pd.read_csv('LFM/test_movielens.csv'))
