import os
import pandas as pd
import numpy as np
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, dot, Dense, Concatenate, Add  # noqa
from keras.models import load_model
from keras import optimizers
import tensorflow as tf
from base.Model import Model
from base.Item import Item
from base.User import User

EMBEDDING_DIM = 200

class LFM(Model):
    def __init__(self, data_type, n, neg_frac_in_train, merge_type="dot", ensure_new=True):
        super().__init__(n, "LFM", data_type, ensure_new)
        self.name += "_neg_{}_{}".format(neg_frac_in_train, merge_type)
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
        dense_1 = Dense(units=128, input_shape=(EMBEDDING_DIM, ),
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
        dense_1 = Dense(units=128, input_shape=(EMBEDDING_DIM*2, ),
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
        # struct for item, input for embedding layer must be int
        item_input = Input(shape=[1], name='Item')
        item_embed = Embedding(input_dim=self.max_item_id+1,
                               output_dim=EMBEDDING_DIM,
                               input_length=1,
                               name='item_embedding')(item_input)
        # need to flatten even when matrix has only one row
        item_vec = Flatten(name='item_flatten')(item_embed)
        # struct for user
        user_input = Input(shape=[1], name='User')
        user_embed = Embedding(input_dim=self.max_user_id+1,
                               output_dim=EMBEDDING_DIM,
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
        optimizer = optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer,
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        keras.utils.plot_model(self.model,
                               to_file='models/model_struc/model_{}.png'.format(self.merge_type),
                               show_shapes=True, show_layer_names=True)
        self.model.summary()

    def fit(self, samples):
        # record user's history items in the training data
        assert len(pd.unique(samples['event'])) == 2
        events = samples.loc[samples['event'] == 1, :]
        if super().fit(events):
            return
        del events
        # try to load previous trained model
        try:
            self.model = load_model("models/saved_models/{}.h5".format(self.name))
            return
        except OSError:
            print("[{}] Previous model not found, train a new model".format(self.name))
        # convert id to int for embedding layers
        self.max_user_id = max(samples['visitorid'])
        self.max_item_id = max(samples['itemid'])
        self.construct_model()
        self.train(samples)
        self.save()

    def train(self, train_data):
        self.model.fit([train_data['itemid'],
                        train_data['visitorid']],
                       train_data['event'],
                       epochs=30)

    def evaluate_prediction(self, test_data):
        # input order must corresponding to the model input building order
        result = self.model.evaluate(x=[test_data['itemid'],
                                        test_data['visitorid']],
                                     y=np.array([1 for _ in range(len(test_data))]).reshape(len(test_data), 1))
        print(result)

    def make_recommendation(self, user_id):
        """
            use batches to predict user's interest to all items
            much faster than predict one sample at a time
        """
        try:
            user = self.users[user_id]
        except KeyError:
            print('User {} not shown in the training set.'.format(user_id))
            return -1
        history_items = user.covered_items
        samples = []
        for item_id, item in self.items.items():
            if item_id in history_items:
                continue
            samples.append([item_id, user_id])
        # prepare inputs (list of itemid array and userid array) to predict
        # array's shape is (n_samples * n_values_persample)
        itemid_input = np.array([sample[0] for sample in samples]).reshape(len(samples), 1)
        userid_input = np.array([sample[1] for sample in samples]).reshape(len(samples), 1)
        inputs = [itemid_input, userid_input]
        # make prediction
        interests = self.model.predict(inputs, batch_size=128)
        items_rank = {}
        for item_id, interest in zip(itemid_input, interests):
            items_rank[item_id[0]] = interest[0]
        reco_items = super().get_top_n_items(items_rank)
        return reco_items

    def evaluate(self, test_data):
        # convert id to int
        # self.evaluate_prediction(test_data)
        return super().evaluate_recommendation(test_data)

    def save(self):
        super().save()
        keras_model = os.path.join('models/saved_models/keras_model_{}'.format(self.name + '.h5'))
        self.model.save(keras_model)
        print("[{}] Model saved".format(self.name))

    def load(self):
        super().load()
        keras_model = os.path.join('models/saved_models/keras_model_{}'.format(self.name + '.h5'))
        self.model = load_model(keras_model)
        print("[{}] Previous keras model found and loaded.".format(self.name))