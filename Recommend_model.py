import pandas as pd
import pickle
import os


class Item(object):
    def __init__(self, id):
        assert isinstance(id, int)
        self.id = id
        self.covered_users = set()


class User(object):
    def __init__(self, id):
        assert isinstance(id, int)
        self.id = id
        self.covered_items = set()


class Recommend_model:
    def __init__(self, n, model_type, data_type, ensure_new=True):
        """base class for all recommendation models

        Parameters
        ----------
        n : [int]
            [number of items to be recommended to one user]
        model_type : [str]
            [model's name, e.g. item CF]
        """
        self.n = n
        self.model_type = model_type
        self.data_type = data_type
        self.name = 'model_{}_{}_n_{}'.format(data_type, model_type, n)
        self.ensure_new = ensure_new

    def init_item_and_user_objects(self, event_data):
        """
            iter through the training data, doing:
                Init Item object for each item,
                and record unique users who touched this item
                Init User object for each user,
                and record unique items touched by the user
        """
        if len(self.users) != 0 or len(self.items) != 0:
            raise ValueError('Item and user objects already been inited!')
        for index, row in event_data.iterrows():
            # find if the item and user has been created
            item_id, user_id = int(row['itemid']), int(row['visitorid'])
            try:
                self.items[item_id].covered_users.add(user_id)
            except KeyError:
                item = Item(item_id)
                item.covered_users.add(user_id)
                self.items[item_id] = item
            try:
                self.users[user_id].covered_items.add(item_id)
            except KeyError:
                user = User(user_id)
                user.covered_items.add(item_id)
                self.users[user_id] = user

    def fit(self, train_data, force_training=False):
        """Init user and item dict,
           self.users -> {user_id:user_object}
           self.item -> {item_id:item_object}
        """
        if not force_training:
            try:
                self.load()
                return "previous model loaded"
            except OSError:
                print("[{}] Previous trained model not found, start training a new one...".format(self.name))  # noqa
        print("[{}] Init user and item objects...".format(self.name))
        self.users, self.items = {}, {}
        self.init_item_and_user_objects(train_data)
        print("[{}] Init done!".format(self.name))

    def get_top_n_items(self, items_rank):
        items_rank = sorted(items_rank.items(), key=lambda item: item[1],
                            reverse=True)
        items_id = [x[0] for x in items_rank]
        if len(items_id) < self.n:
            print("Number of ranked items is smaller than n:{}".format(self.n))
            return items_id
        return items_id[:self.n]

    def valid_user(self, user_id):
        if user_id in self.users.keys():
            return True
        print("[{}] User {} not seen in the training set.".format(self.name, user_id))  # noqa
        return False

    def compute_n_hit(self, reco_items_id, real_items_id):
        """Count common items between one user's real items and recommend items
        """
        n_hit = 0
        for item_id in real_items_id:
            if item_id in reco_items_id:
                n_hit += 1
        return n_hit

    @staticmethod
    def get_user_real_items(user_id, test_data):
        # get user's real interested items id
        boolIndex = test_data['visitorid'] == user_id
        user_data = test_data.loc[boolIndex, :]
        real_items_id = pd.unique(user_data['itemid'])
        return real_items_id

    def evaluate_recommendation(self, test_data):
        """compute average recall, precision and coverage upon test data
        """
        print("[{}] Start evaluating model with test data...".format(self.name))  # noqa
        users_id = pd.unique(test_data['visitorid']).astype(int)
        recall = precision = n_valid_users = covered_users = 0
        covered_items = set()
        for user_id in users_id:
            real_items_id = self.get_user_real_items(user_id, test_data)
            reco_items_id = self.make_recommendation(user_id)
            if not isinstance(reco_items_id, list):
                print('[{}] Cannot make recommendation for user {}'.format(self.name, user_id))  # noqa
                continue
            n_hit = self.compute_n_hit(reco_items_id, real_items_id)
            # recall
            recall += n_hit/len(real_items_id)
            # precision
            precision += n_hit/len(reco_items_id)
            # coverage
            covered_items.update(reco_items_id)
            n_valid_users += 1
        recall /= n_valid_users
        precision /= n_valid_users
        coverage = len(covered_items)/len(self.items.keys())
        print('[{}] Number of valid unique users: {}'.format(self.name, n_valid_users))  # noqa
        print('[{}] Total unique users in the test set: {}'.format(self.name, len(pd.unique(test_data['visitorid']))))  # noqa
        print('[{}] Recall:{}, Precision:{}, Coverage:{}'.format(self.name, recall, precision, coverage))  # noqa
        return {'recall': recall, 'precision': precision, 'coverage': coverage}

    def save(self):
        file_path = os.path.join('saved_models', self.name + '.pickle')
        with open(file_path, 'wb') as f:
            f.write(pickle.dumps(self))
        print("[{}] Model saved to {}".format(self.name, file_path))

    def load(self):
        print("[{}] Trying to find and load previous trained model...".format(self.name))  # noqa
        file_path = os.path.join('saved_models', self.name + '.pickle')
        with open(file_path, 'rb') as f:
            # self = pickle.loads(f.read())  # reference detached!
            self.__dict__.update(pickle.loads(f.read()).__dict__)
        print("[{}] Previous trained model found and loaded.".format(self.name))  # noqa
