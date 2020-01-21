import pandas as pd
import pickle
import os
from math import sqrt
from .user_distance import compute_distance
from .user import User
from .item import Item


class UserCF(object):
    def __init__(self, all_user_id, k):
        """pass unique id for users and items
        """
        self.users, self.items = {}, {}
        # init 'matrix', using dict rather than list of list
        # so that the user_id is not ristricted to be 0~len(users)-1
        self.user_sim_matrix = {int(user_id): {} for user_id in all_user_id}
        self.k = k

    def form_item_objects(self, event_data):
        """init item object for each item, and record
           unique users who touched this item
           count user's unique items at the same time
        """
        for index, row in event_data.iterrows():
            user_id, item_id = int(row['visitorid']), int(row['itemid'])
            # find if the item and user has been created
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

    def update_user_count_dict(self, user_A_id, user_B_id):
        """find user_A's count dict in the sim matrix,
           update the dict by add 1 to the value of user_B
        """
        try:
            count_dict = self.user_sim_matrix[user_A_id]
        except KeyError:
            count_dict = self.user_sim_matrix[user_A_id] = {}  # all reference
        try:
            count_dict[user_B_id] += 1
        except KeyError:
            count_dict[user_B_id] = 1

    def form_users_distance_matrix(self, event_data, metric):
        # only consider 'transaction' and 'add to chart' action, which is 3 or 2
        bool_index = event_data['event'] != 1
        event_data = event_data.loc[bool_index, :]
        self.form_item_objects(event_data)
        # update user-user common items counts
        for item in self.items.values():
            users = list(item.covered_users)  # convert to list for indexing
            # iter through all user pairs
            for i in range(len(users)-1):
                for j in range(i+1, len(users)):
                    user_A_id, user_B_id = users[i], users[j]
                    self.update_user_count_dict(user_A_id, user_B_id)
                    self.update_user_count_dict(user_B_id, user_A_id)
        # divide counts
        for user_id, count in self.user_sim_matrix.items():  # count is reference
            try:
                all_count = len(self.users[user_id].covered_items)
            except KeyError:
                # the user which comes from the full set,
                # not in test set, so make it remain empty
                continue
            for another_user_id in count.keys():
                another_user = self.users[another_user_id]
                all_count_another = len(another_user.covered_items)
                count[another_user_id] /= sqrt(all_count*all_count_another)
                assert count[another_user_id] <= 1

    def save(self, dir):
        with open(os.path.join(dir, 'users.pickle'), 'wb') as f:
            f.write(pickle.dumps(self.users))
        with open(os.path.join(dir, 'items.pickle'), 'wb') as f:
            f.write(pickle.dumps(self.items))
        with open(os.path.join(dir, 'user_sim_matrix.pickle'), 'wb') as f:
            f.write(pickle.dumps(self.user_sim_matrix))

    def load(self, dir):
        with open(os.path.join(dir, 'users.pickle'), 'rb') as f:
            self.users = pickle.loads(f.read())
        with open(os.path.join(dir, 'items.pickle'), 'rb') as f:
            self.items = pickle.loads(f.read())
        with open(os.path.join(dir, 'user_sim_matrix.pickle'), 'rb') as f:
            self.user_sim_matrix = pickle.loads(f.read())

    def rank_potential_items(self, target_user_id, related_users_id):
        items_rank = {}
        target_user = self.users[target_user_id]
        for user_id in related_users_id:
            similar_user = self.users[user_id]
            similarity = self.user_sim_matrix[target_user_id][user_id]
            for item_id in similar_user.covered_items:
                if item_id in target_user.covered_items:
                    continue
                score = similarity * 1
                try:
                    items_rank[item_id] += score
                except KeyError:
                    items_rank[item_id] = score
        return items_rank

    def make_recommendation(self, user_id):
        try:
            target_user = self.users[user_id]
            # find the top k users that most like the input user
            related_users = self.user_sim_matrix[user_id]
            if len(related_users) == 0:
                # print('user {} didn\'t has any common item with other users')
                return -1
            related_users = sorted(related_users.items(),
                                   key=lambda item: item[1],
                                   reverse=True)  # return a list of tuples
            if len(related_users) >= self.k:
                related_users_id = [x[0] for x in related_users[:self.k]]
            else:
                related_users_id = [x[0] for x in related_users]
            return self.rank_potential_items(user_id, related_users_id)
        except KeyError:
            # print('User {} has not shown in the training set.')
            return -2

    def make_prediction(self, user_id, item_id):
        items = self.make_recommendation(user_id)
        if isinstance(items, dict):
            if item_id in items.keys():
                return 1
            return 0
        elif items == -2:
            return -2
        elif items == -1:
            return 0

    def evaluate(self, test_data, metric):
        if metric == 'prediction':
            i, correct = 0, 0
            for index, row in test_data.iterrows():
                flag = self.make_prediction(row['visitorid'], row['itemid'])
                if flag < 0:
                    continue
                correct += flag
                i += 1
        print('prediction precision: {}'.format(correct/i))

