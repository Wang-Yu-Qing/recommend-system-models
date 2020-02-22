import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from utils.Feature_util import Feature_util
from base.Model import Model


class Wide_and_deep(Model):
    def __init__(self, n, data_type, neg_frac_in_train, ensure_new=True):
        super().__init__(n, "Wide&Deep", data_type, ensure_new=ensure_new)
        self.name += "_neg_{}".format(neg_frac_in_train)
        # keys to get input layers in all input layers dict
        self.deep_inputs = ["visitorid", "age", "zip_code", "gender", "occupation", "itemid", "release_date"]
        self.wide_inputs = ["gender", "occupation", "unknown", "action", "adventure", "animation",
                            "child", "comedy", "crime", "docu", "drama", "fantasy", "noir", "horror",
                            "musical", "mystery", "romance", "sci", "thrill", "war", "western"]
        # keys to get feature columns in all feature columns dict
        self.deep_features = ["visitorid", "age", "zip_code", "gender", "occupation", "itemid", "release_date"]
        self.wide_features = ["gender_x_occupation", "cate_x_cate"]
        self.item_info_map = {}
        self.user_info_map = {}

    @staticmethod
    def df_to_dataset(dataframe, shuffle=False, batch_size=128):
        assert type(dataframe) == pd.DataFrame
        dataframe = dataframe.copy()
        labels = dataframe.pop('event')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    def build_model(self, users_info, items_info):
        FU = Feature_util(self.data_type)
        # 1. define input layer
        input_layer = FU.create_movie_lens_input_layer()
        # 2. prepare feature columns (list of feature column objects)
        FU.create_movie_lens_user_feature_columns(users_info)
        FU.create_movie_lens_item_feature_columns(items_info)
        all_fc = FU.data_map["MovieLens_100K"]["feature_columns"]
        # 3. select wide features and deep features, convert feature columns to feature layer
        deep_fc = [all_fc[fc_name] for fc_name in self.deep_features]
        wide_fc = [all_fc[fc_name] for fc_name in self.wide_features]
        deep_feature_layer = layers.DenseFeatures(deep_fc, name="deep_feature_layer")
        wide_feature_layer = layers.DenseFeatures(wide_fc, name="wide_feature_layer")
        # 4. convert input vector to feature vector 
        #    feature columns' key and input layer dict's key corresponds
        deep_feature_vec = deep_feature_layer({input_name: input_layer[input_name] 
                                               for input_name in self.deep_inputs})
        wide_feature_vec = wide_feature_layer({input_name: input_layer[input_name]
                                               for input_name in self.wide_inputs})
        # 5. network structure for deep model
        deep_1 = layers.Dense(512, activation="relu", name="deep_1")(deep_feature_vec)
        deep_2 = layers.Dense(256, activation="relu", name="deep_2")(deep_1)
        deep_3 = layers.Dense(128, activation="relu", name="deep_3")(deep_2)
        deep_4 = layers.Dense(64, activation="relu", name="deep_4")(deep_3)
        # 6. concatenate with wide feature
        concat = layers.concatenate([deep_4, wide_feature_vec], name="concat")
        # 7. define output layer
        output = layers.Dense(1, activation="sigmoid", name="output")(concat)
        # 8. build and compile model
        self.model = tf.keras.Model(inputs=[v for v in input_layer.values()], outputs=output)
        self.model.compile(optimizer="adam",
                           loss='binary_crossentropy',
                           metrics=["accuracy"])
        tf.keras.utils.plot_model(self.model,
                                  to_file='models/model_struc/wide&deep.png',
                                  show_shapes=True, show_layer_names=True)
        self.model.summary()

    @staticmethod
    def input_check(train_data):
        assert not train_data.isnull().values.any()

    def init_info_map(self, users_info, items_info):
        """
            init user and item info map for prediction
        """
        user_info_seq = Feature_util.data_map["MovieLens_100K"]["user_info_seq"]
        item_info_seq = Feature_util.data_map["MovieLens_100K"]["item_info_seq"]
        for index, row in users_info.iterrows():
            # make sure order of row's values correspond to input values
            self.user_info_map[row["visitorid"]] = list(row[user_info_seq])
        for index, row in items_info.iterrows():
            self.item_info_map[row["itemid"]] = list(row[item_info_seq])

    def fit(self, train_data, users_info, items_info):
        # user positive samples to generate history records
        positive_samples = train_data.loc[train_data["event"] == 1, ("visitorid", "itemid", "timestamp")]
        if super().fit(positive_samples):
            return
        del positive_samples
        self.init_info_map(users_info, items_info)
        # build model
        self.build_model(users_info, items_info)
        del train_data["rating"], train_data["timestamp"]
        # MAKE SURE col values' type identity with input layer
        # MAKE SURE columns' order identity with input order, label comes last
        train_data = train_data[Feature_util.data_map["MovieLens_100K"]["input_values_seq"]]
        self.input_check(train_data)
        # TODO: not convert to dataset?
        train_data = self.df_to_dataset(train_data)
        self.model.fit(train_data, epochs=30)
        self.save()

    def make_recommendation(self, user_id):
        """
            use batches to predict user's interest to all items
            much faster than predict one sample at a time
        """
        # get user info
        user_info = self.user_info_map[user_id]
        history_items = self.users[user_id].covered_items
        # prepare all samples to predict
        samples = []
        for item_id, item in self.items.items():
            if item_id in history_items:
                continue
            # get item info
            item_info = self.item_info_map[item_id]
            # input values order is guaranteed in self.init_info_map()
            sample = user_info + item_info
            samples.append(sample)
        if len(samples) == 0:  # all items have been touched
            return -1
        # convert input samples to list of arrays (n_inputs * (n_samples * n_values_per_sample))
        n_inputs, n_samples, n_values = len(sample), len(samples), 1
        inputs = []
        for i in range(len(sample)):
            input_array = np.array([sample[i] for sample in samples]).reshape(n_samples, n_values)
            inputs.append(input_array)
        # make prediction
        interests = self.model.predict(inputs, batch_size=128)
        # get items id's input array
        items_id = inputs[5]
        items_rank = {}
        for item_id, interest in zip(items_id, interests):
            items_rank[item_id[0]] = interest[0]
        return super().get_top_n_items(items_rank)

    def evaluate(self, test_data):
        # make sure test_data row values order correct
        # self.evaluate_prediction(test_data)
        return self.evaluate_recommendation(test_data)

    def evaluate_prediction(self, test_data):
        test_data = self.df_to_dataset(test_data)
        print(self.model.evaluate(test_data))

    def load_keras_model(self):
        try:
            self.model = load_model("models/saved_models/{}.h5".format(self.name))
            print("[{}] Previous trained model loaded.".format(self.name))
            return 0
        except OSError:
            return -1

    def save(self):
        super().save()
        keras_model = os.path.join('models/saved_models/keras_model_{}'.format(self.name + '.h5'))
        self.model.save(keras_model)
        user_info_map = os.path.join("models/saved_models/user_info_{}".format(self.name + ".pickle"))
        with open(user_info_map, "wb") as f:
            f.write(pickle.dumps(self.user_info_map))
        item_info_map = os.path.join("models/saved_models/item_info_{}".format(self.name + ".pickle"))
        with open(item_info_map, "wb") as f:
            f.write(pickle.dumps(self.item_info_map))
        print("[{}] Model saved".format(self.name))

    def load(self):
        super().load()
        user_info_map = os.path.join("models/saved_models/user_info_{}".format(self.name + ".pickle"))
        with open(user_info_map, "rb") as f:
            self.user_info_map = pickle.loads(f.read())
        item_info_map = os.path.join("models/saved_models/item_info_{}".format(self.name + ".pickle"))
        with open(item_info_map, "rb") as f:
            self.item_info_map = pickle.loads(f.read())
        keras_model = os.path.join('models/saved_models/keras_model_{}'.format(self.name + '.h5'))
        self.model = load_model(keras_model)
        print("[{}] Previous keras model found and loaded.".format(self.name))