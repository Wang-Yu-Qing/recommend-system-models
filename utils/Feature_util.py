import time
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from itertools import islice
from .Data_util import Data_util

EMBEDDING_DIM = 200

class Feature_util:
    data_map = {
        "MovieLens_100K": {
            "user_info": "data/MovieLens_100K/u.user",
            "item_info": "data/MovieLens_100K/u.item",
            "event_data": "data/MovieLens_100K/ratings.dat",
            "skip_col_names": False,
            "sep": "|",
            "user_col_names": ["visitorid", "age", "gender", "occupation", "zip_code"],
            "item_col_names": ["itemid", "title", "release_date", "video_release_date",
                               "URL", "unknown", "action", "adventure", "animation", "child",
                               "comedy", "crime", "docu", "drama", "fantasy", "noir", "horror",
                               "musical", "mystery", "romance", "sci", "thrill", "war", "western"],
            # input user and item values order for test data
            "user_info_seq": ["visitorid", "age", "zip_code", "gender", "occupation"],
            "item_info_seq": ["itemid", "release_date", "unknown", "action", "adventure",
                              "animation", "child", "comedy", "crime", "docu", "drama",
                              "fantasy", "noir", "horror", "musical", "mystery", "romance",
                              "sci", "thrill", "war", "western"],
            # input values order for train data
            "input_values_seq": ["visitorid", "age", "zip_code", "gender", "occupation",
                                  "itemid", "release_date", "unknown", "action", "adventure",
                                  "animation", "child", "comedy", "crime", "docu", "drama",
                                  "fantasy", "noir", "horror", "musical", "mystery", "romance",
                                  "sci", "thrill", "war", "western", "event"],
            "feature_columns": {"visitorid": None, "age":None, "zip_code":None, "gender": None,
                                "occupation": None, "gender_x_occupation":None, "itemid": None,
                                "release_date": None, "cate_x_cate": None}
        }
    }

    def __init__(self, data_type):
        self.data_type = data_type

    def read_info_file(self, info_type):
        info_path = self.data_map[self.data_type][info_type+"_info"]
        columns = self.data_map[self.data_type][info_type+"_col_names"]
        skip_col_names = self.data_map[self.data_type]["skip_col_names"]
        sep = self.data_map[self.data_type]["sep"]
        with open(info_path, "r") as f:
            if skip_col_names:
                f.readline()
            lines = [Data_util.parse_line(line, sep) for line in islice(f, None)]
        return pd.DataFrame(lines, columns=columns)
   
    def read_user_item_info(self):
        users_info = self.read_info_file("user")
        items_info = self.read_info_file("item")
        # convert data type
        users_info["age"] = users_info["age"].apply(lambda x : int(x))
        users_info["visitorid"] = users_info["visitorid"].apply(lambda x : int(x))
        items_info["itemid"] = items_info["itemid"].apply(lambda x : int(x))
        # release time to timestamp normalized
        items_info.loc[:, "release_date"] = items_info["release_date"].apply(Feature_util.datetime_parser)
        # repalce except value with average timestamp
        avg_timestamp = items_info["release_date"].mean(skipna=True)
        items_info.loc[pd.isnull(items_info["release_date"]), "release_date"] = avg_timestamp
        assert sum(pd.isnull(items_info["release_date"])) == 0
        # normalize timestamp
        std_timestamp = items_info["release_date"].std()
        items_info.loc[:, "release_date"] = (items_info["release_date"]-avg_timestamp)/std_timestamp
        return users_info, items_info

    # A utility method to show transromation from feature column
    @staticmethod
    def demo(feature_column, data):
        """
            data must be a dictionary
        """
        feature_layer = layers.DenseFeatures(feature_column)
        column_values = feature_layer(data).numpy()
        print("feature dimension: ", len(column_values[0]))
        print(column_values)

    def create_movie_lens_user_feature_columns(self, samples):
        feature_columns = self.data_map["MovieLens_100K"]["feature_columns"]
        # dict type of data
        samples_dict = samples.to_dict("list")
        # vocabulary lists
        gender_list = list(pd.unique(samples["gender"]))
        occupation_list = list(pd.unique(samples["occupation"]))
        # number of categories of each column
        n_occupation = len(pd.unique(samples["occupation"]))
        n_gender = len(pd.unique(samples["gender"]))
        n_users = len(pd.unique(samples["visitorid"]))
        n_zip_code = len(pd.unique(samples["zip_code"]))
        # visitorid -> embedding for deep model
        fc_visitorid = tf.feature_column.categorical_column_with_hash_bucket("visitorid", n_users+1, dtype=tf.int32)
        fc_visitorid = tf.feature_column.embedding_column(fc_visitorid, dimension=EMBEDDING_DIM)
        feature_columns["visitorid"] = fc_visitorid
        # age -> numeric for deep model
        # (corresponding age column values must be numeric not string)
        fc_age = tf.feature_column.numeric_column("age")
        feature_columns["age"] = fc_age
        # zip_code -> embedding for deep model
        fc_zip_code = tf.feature_column.categorical_column_with_hash_bucket("zip_code", n_zip_code+1)
        fc_zip_code = tf.feature_column.embedding_column(fc_zip_code, dimension=EMBEDDING_DIM)
        feature_columns["zip_code"] = fc_zip_code
        # gender -> one hot for deep model
        fc_gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", 
                                                                              vocabulary_list=gender_list)
        fc_gender = tf.feature_column.indicator_column(fc_gender)
        feature_columns["gender"] = fc_gender
        # occupation -> embedding for deep model
        fc_occupation = tf.feature_column.categorical_column_with_vocabulary_list("occupation",
                                                                                  vocabulary_list=occupation_list)
        fc_occupation = tf.feature_column.embedding_column(fc_occupation, dimension=EMBEDDING_DIM)
        feature_columns["occupation"] = fc_occupation
        # gender_x_occupation(crossed feature) -> one hot (full cross, really wide) for wide model
        fc_gender_x_occupation = tf.feature_column.crossed_column(["gender",
                                                                   "occupation"],
                                                                  n_occupation*n_gender+1)
        fc_gender_x_occupation = tf.feature_column.indicator_column(fc_gender_x_occupation)
        feature_columns["gender_x_occupation"] = fc_gender_x_occupation
    
    @staticmethod
    def datetime_parser(value):
        try:
            datetime_obj = datetime.strptime(value, "%d-%b-%Y")
            return time.mktime(datetime_obj.timetuple())
        except ValueError as E:
            print("Wrong datetime value: {}, return na".format(value))
            return None
    
    def create_movie_lens_item_feature_columns(self, samples):
        feature_columns = self.data_map["MovieLens_100K"]["feature_columns"]
        # dict type of data
        samples_dict = samples.to_dict("list")
        # vocabulary lists
        # number of categories of each column
        n_items = len(pd.unique(samples["itemid"]))
        # item_id -> embedding for deep model
        fc_itemid = tf.feature_column.categorical_column_with_hash_bucket("itemid", n_items+1, dtype=tf.int32)
        fc_itemid = tf.feature_column.embedding_column(fc_itemid, dimension=EMBEDDING_DIM)
        feature_columns["itemid"] = fc_itemid
        # release_date -> numeric
        fc_release_date = tf.feature_column.numeric_column("release_date")
        feature_columns["release_date"] = fc_release_date
        # categories -> crossed, allow some hash conflicts, for wide model
        # TODO: user raw 0/1 categories as input feature vector?
        fc_cate_x_cate = tf.feature_column.crossed_column(['unknown', 'action', 'adventure', 'animation', 'child', 'comedy',
                                                           'crime', 'docu', 'drama', 'fantasy', 'noir', 'horror', 'musical',
                                                           'mystery', 'romance', 'sci', 'thrill', 'war', 'western'],
                                                           1500)
        fc_cate_x_cate = tf.feature_column.indicator_column(fc_cate_x_cate)
        feature_columns["cate_x_cate"] = fc_cate_x_cate
    
    def create_movie_lens_input_layer(self):
        input_layer = {}
        # key must correspond to the key in feature columns,
        # data type should identity with feature columns
        # input orders remains in the dict, should be same as the input row's values
        # user input
        input_layer["visitorid"] = tf.keras.Input(shape=(1,), name="visitorid", dtype=tf.int32)
        input_layer["age"] = tf.keras.Input(shape=(1,), name="age", dtype=tf.int32)
        input_layer["zip_code"] = tf.keras.Input(shape=(1,), name="zip_code", dtype=tf.string)
        input_layer["gender"] = tf.keras.Input(shape=(1,), name="gender", dtype=tf.string)
        input_layer["occupation"] = tf.keras.Input(shape=(1,), name="occupation", dtype=tf.string)
        # item input
        input_layer["itemid"] = tf.keras.Input(shape=(1,), name="itemid", dtype=tf.int32)
        input_layer["release_date"] = tf.keras.Input(shape=(1,), name="release_date", dtype=tf.float64)
        input_layer["unknown"] = tf.keras.Input(shape=(1,), name="unknown", dtype=tf.string)
        input_layer["action"] = tf.keras.Input(shape=(1,), name="action", dtype=tf.string)
        input_layer["adventure"] = tf.keras.Input(shape=(1,), name="adventure", dtype=tf.string)
        input_layer["animation"] = tf.keras.Input(shape=(1,), name="animation", dtype=tf.string)
        input_layer["child"] = tf.keras.Input(shape=(1,), name="child", dtype=tf.string)
        input_layer["comedy"] = tf.keras.Input(shape=(1,), name="comedy", dtype=tf.string)
        input_layer["crime"] = tf.keras.Input(shape=(1,), name="crime", dtype=tf.string)
        input_layer["docu"] = tf.keras.Input(shape=(1,), name="docu", dtype=tf.string)
        input_layer["drama"] = tf.keras.Input(shape=(1,), name="drama", dtype=tf.string)
        input_layer["fantasy"] = tf.keras.Input(shape=(1,), name="fantasy", dtype=tf.string)
        input_layer["noir"] = tf.keras.Input(shape=(1,), name="noir", dtype=tf.string)
        input_layer["horror"] = tf.keras.Input(shape=(1,), name="horror", dtype=tf.string)
        input_layer["musical"] = tf.keras.Input(shape=(1,), name="musical", dtype=tf.string)
        input_layer["mystery"] = tf.keras.Input(shape=(1,), name="mystery", dtype=tf.string)
        input_layer["romance"] = tf.keras.Input(shape=(1,), name="romance", dtype=tf.string)
        input_layer["sci"] = tf.keras.Input(shape=(1,), name="sci", dtype=tf.string)
        input_layer["thrill"] = tf.keras.Input(shape=(1,), name="thrill", dtype=tf.string)
        input_layer["war"] = tf.keras.Input(shape=(1,), name="war", dtype=tf.string)
        input_layer["western"] = tf.keras.Input(shape=(1,), name="western", dtype=tf.string)
        assert list(input_layer.keys()) == self.data_map["MovieLens_100K"]["input_values_seq"][:-1]
        return input_layer