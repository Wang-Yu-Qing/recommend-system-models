import pandas as pd
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, dot, Dense, Concatenate, Add  # noqa
from keras.models import load_model
from keras import optimizers
from Recommend_model import Recommend_model, User, Item


class LFM(Recommend_model):
    def __init__(self, data_type, n, hidden_dim,
                 neg_frac_in_train, merge_type="dot", ensure_new=True):
        super().__init__(n, "LFM", data_type, ensure_new)
        self.name += "_neg_{}_{}_dim_{}".format(neg_frac_in_train, merge_type, hidden_dim)  # noqa
        self.hidden_dim = hidden_dim
        self.n = n
        self.ensure_new = ensure_new
        self.merge_type = merge_type

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

    def construct_model(self):
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
        if self.merge_type == 'dot':
            out_put = self.dot_structure(item_vec, user_vec)
        elif self.merge_type == 'add':
            out_put = self.adding_structure(item_vec, user_vec)
        elif self.merge_type == 'concat':
            out_put = self.concat_structure(item_vec, user_vec)
        else:
            raise ValueError('Invalid embeds merge type provided!')
        self.model = keras.Model([item_input, user_input], out_put)
        optimizer = optimizers.Adam(lr=0.0005)
        self.model.compile(optimizer, 'mse')
        keras.utils.plot_model(self.model,
                               to_file='LFM/model_struc/model_{}.png'.format(
                                   self.merge_type),
                               show_shapes=True, show_layer_names=True)
        self.model.summary()

    def fit(self, samples, force_training, save=True):
        # record user's history items in the training data
        assert len(pd.unique(samples['event'])) == 2
        pos_samples = samples.loc[samples['event'] == 1, :]
        super().fit(pos_samples)
        self.max_user_id = max(samples['visitorid'])
        self.max_item_id = max(samples['itemid'])
        self.construct_model()
        self.train(samples)
        if save:
            super().save()

    def train(self, train_data):
        self.model.fit([train_data['itemid'],
                        train_data['visitorid']],
                       train_data['event'],
                       epochs=20)

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
        return reco_items

    def evaluate(self, test_data):
        self.evaluate_mse(test_data)
        super().evaluate_recommendation(test_data)
