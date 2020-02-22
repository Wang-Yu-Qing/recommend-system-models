import pandas as pd
import pickle
import os
from .User import User
from .Item import Item
from .Tag import Tag


class Model:
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
        self.name = '{}_{}'.format(data_type, model_type)
        self.ensure_new = ensure_new

    @staticmethod
    def time_elapse(t1, t2, alpha=0.5):
        return 1/(1+alpha*abs(t1-t2))

    @staticmethod
    def update_obj(dic, obj_id, timestamp):
        """if timestamp is considered, update the
           object(item/user) in the dict(covered_items/covered_users)
           with the latest timestamp
        """
        try:
            _timestamp = dic[obj_id]
            if timestamp < _timestamp:
                return
        except KeyError:
            pass
        dic[obj_id] = timestamp

    @staticmethod
    def update_tag(tag_id, tags, user, item):
        try:
            tag = tags[tag_id]
        except KeyError:
            tag = Tag(tag_id)
            tags[tag_id] = tag
        tag.n_used += 1
        # update tag's item count
        try:
            tag.items_count[item.id] += 1
        except KeyError:
            tag.items_count[item.id] = 1
        # update user's tag count
        try:
            user.tags_count[tag_id] += 1
        except KeyError:
            user.tags_count[tag_id] = 1
        # update item's tag count
        try:
            item.tags_count[tag_id] += 1
        except KeyError:
            item.tags_count[tag_id] = 1

    @staticmethod
    def init_item_and_user_objects(event_data, tag=False):
        """
            iter through the training data, doing:
                Init Item object for each item,
                and record unique users who touched this item
                Init User object for each user,
                and record unique items touched by the user
        """
        items, users = {}, {}
        if tag:
            tags = {}
        for index, row in event_data.iterrows():
            item_id, user_id = int(row['itemid']), int(row['visitorid'])
            event_time = row['timestamp']
            item_info, user_info = (item_id, event_time), (user_id, event_time)  # noqa
            # find if the item and user has been created
            try:
                Model.update_obj(items[item_id].covered_users,
                                 user_id, event_time)
            except KeyError:
                item = Item(item_id)
                Model.update_obj(item.covered_users,
                                 user_id, event_time)
                items[item_id] = item
            try:
                Model.update_obj(users[user_id].covered_items,
                                 item_id, event_time)
            except KeyError:
                user = User(user_id)
                Model.update_obj(user.covered_items,
                                 item_id, event_time)
                users[user_id] = user
            if tag:
                user, item = users[user_id], items[item_id]
                # all reference
                Model.update_tag(row["tagid"], tags, user, item)
        if tag:
            # sort tag's items count dict
            for tag in tags.values():
                tag.items_count = dict(sorted(tag.items_count.items(),
                                              key=lambda item: item[1],
                                              reverse=True))
            return items, users, tags
        return items, users

    def fit(self, train_data, tag=False):
        """Init user and item dict,
           self.users -> {user_id:user_object}
           self.item -> {item_id:item_object}
        """
        try:
            self.load()
            return 1
        except OSError as E:
            print(E)
            print("[{}] Previous trained model not found, start forming history info...".format(self.name))
        print("[{}] Init user and item objects...".format(self.name))
        if tag:
            self.items, self.users, self.tags = self.init_item_and_user_objects(train_data, tag)
        else:
            self.items, self.users = self.init_item_and_user_objects(train_data)
        print("[{}] Init done!".format(self.name))

    def get_top_n_items(self, items_rank):
        items_rank = sorted(items_rank.items(), key=lambda item: item[1],
                            reverse=True)
        items_id = [x[0] for x in items_rank]
        if len(items_id) < self.n:
            print("Number of ranked items is smaller than n:{}".format(self.n))
        return set(items_id[:self.n])  # further lookup complexity is O(1)

    def valid_user(self, user_id):
        if user_id in self.users.keys():
            return True
        print("[{}] User {} not seen in the training set.".format(self.name, user_id))
        return False

    def compute_n_hit(self, reco_items_id, real_items_id):
        """Count common items between one user's real items and recommend items
        """
        n_TP = 0
        for item_id in real_items_id:
            if item_id in reco_items_id: # O(1) complexity for set
                n_TP += 1
        return n_TP

    @staticmethod
    def get_user_real_items(user_id, test_data):
        # get user's real interested items id (unique)
        boolIndex = test_data['visitorid'] == user_id
        user_data = test_data.loc[boolIndex, :]
        real_items_id = pd.unique(user_data['itemid'])
        return real_items_id

    def evaluate_recommendation(self, test_data):
        """compute average recall, precision and coverage upon test event data
        """
        print("[{}] Start evaluating model with test data...".format(self.name))  # noqa
        users_id = pd.unique(test_data['visitorid'])
        recall = precision = n_valid_users = covered_users = fallout = 0
        n_items = len(self.items)
        covered_items = set()
        for user_id in users_id:
            real_items_id = self.get_user_real_items(user_id, test_data)
            reco_items_id = self.make_recommendation(user_id)
            if not isinstance(reco_items_id, set):
                print('[{}] Cannot make recommendation for user {}'.format(self.name, user_id))  # noqa
                continue
            n_TP = self.compute_n_hit(reco_items_id, real_items_id)
            n_FP = len(reco_items_id) - n_TP
            # recall
            recall += n_TP/len(real_items_id)
            # precision
            precision += n_TP/len(reco_items_id)
            # fallout
            fallout += n_FP/(n_items-len(real_items_id))
            # coverage
            covered_items.update(reco_items_id)
            n_valid_users += 1
        recall /= n_valid_users
        precision /= n_valid_users
        fallout /= n_valid_users
        coverage = len(covered_items)/len(self.items)
        print('[{}] Number of valid unique users: {}'.format(self.name, n_valid_users))
        print('[{}] Total unique users in the test set: {}'.format(self.name, len(pd.unique(test_data['visitorid']))))
        print('[{}] Recall:{}, Precision:{}, Coverage:{}'.format(self.name, recall, precision, coverage))
        return {'recall': recall, 'precision': precision, 'fallout': fallout, 'coverage': coverage}

    def save(self):
        """
            all models need to save users and items history info
        """
        users = os.path.join('models/saved_models/users_{}'.format(self.name + '.pickle'))
        items = os.path.join('models/saved_models/items_{}'.format(self.name + '.pickle'))
        with open(users, 'wb') as f:
            f.write(pickle.dumps(self.users))
        with open(items, 'wb') as f:
            f.write(pickle.dumps(self.items))
        print("[{}] users and items objects saved.".format(self.name))

    def load(self):
        """
            all models need to load users and items history info
        """
        print("[{}] Trying to find and load previous history info...".format(self.name))
        users = os.path.join('models/saved_models/users_{}'.format(self.name + '.pickle'))
        items = os.path.join('models/saved_models/items_{}'.format(self.name + '.pickle'))
        with open(users, 'rb') as f:
            self.users = pickle.loads(f.read())
        with open(items, 'rb') as f:
            self.items = pickle.loads(f.read())
        print("[{}] Previous info found and loaded.".format(self.name))
