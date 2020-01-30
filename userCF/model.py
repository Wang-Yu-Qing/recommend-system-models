import pandas as pd
from copy import deepcopy
from math import sqrt, log
from Recommend_model import Recommend_model
from .user import User
from .item import Item


class UserCF(Recommend_model):
    def __init__(self, n, model_type, ensure_new=True):
        super().__init__(n, model_type, ensure_new=ensure_new)
        # init 'similarity matrix', using dict rather than list of list
        # so that the user_id is not ristricted to be 0~len(users)-1
        # for userCF, the matrix keys are user ids.
        # {'userA':{'userB': sim_between_A_and_B, .....},
        #  'userB':{.....},
        #  ....}
        self.sim_matrix = self.users.deepcopy()

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

    def update_user_user_sim(self, user_A_id, user_B_id, item_popularity):
        """when find the user_A and user_B have common item with popularity
           find user_A's row (sim dict) in the sim matrix,
           update the dict by adding the sim value to the value of user_B
        """
        try:
            row = self.sim_matrix[user_A_id]
        except KeyError:
            row = self.sim_matrix[user_A_id] = {}  # all reference
        IIF = log(1+item_popularity)
        try:
            row[user_B_id] += 1/IIF
        except KeyError:
            row[user_B_id] = 1/IIF

    def compute_user_user_sim_base_on_common_items(self):
        """
            for every item, get its covered user pairs
            increase the similarity between the corresponding users in each pair
        """
        for item in self.items.values():
            users = list(item.covered_users)  # convert to list for indexing
            item_popularity = len(users)
            # iter through all user pairs
            for i in range(len(users)-1):
                for j in range(i+1, len(users)):
                    user_A_id, user_B_id = users[i], users[j]
                    # remember to update pair wise!
                    self.update_user_user_sim(user_A_id, user_B_id,
                                              item_popularity)
                    self.update_user_user_sim(user_B_id, user_A_id,
                                              item_popularity)

    def standardize_sim_values(self):
        """
            divide the similarity between A and B by sqrt(lA*lB)
            where lA is the number of unique items A touched and
            lB is the numebr of unique items B touched.
        """
        for user_id_A, row in self.sim_matrix.items():  # count is reference
            lA = len(self.users[user_id_A].covered_items)
            for user_id_B in row.keys():
                lB = len(self.users[user_id_B].covered_items)
                row[user_id_B] /= sqrt(lA*lB)
                assert row[user_id_B] <= 1

    def build_user_user_similarity_matrix(self, event_data):
        """
            form user similarity matrix(dict)
        """
        self.init_item_and_user_objects(event_data)
        self.compute_user_user_sim_base_on_common_items()
        self.standardize_sim_values()

    def fit(self, event_data):
        super().fit(event_data)
        self.build_user_user_similarity_matrix(event_data)

    def rank_potential_items(self, target_user_id, related_users_id):
        """rank score's range is (0, +inf)
        """
        items_rank = {}
        target_user = self.users[target_user_id]
        n_similar_users = 0
        for user_id in related_users_id:
            similar_user = self.users[user_id]
            similarity = self.sim_matrix[target_user_id][user_id]
            for item_id in similar_user.covered_items:
                if self.ensure_new and (item_id in target_user.covered_items):
                    continue  # skip item that already been bought
                score = similarity * 1
                try:
                    items_rank[item_id] += score
                except KeyError:
                    items_rank[item_id] = score
            n_similar_users += 1
            break_condition1 = (n_similar_users >= self.k) and (
                len(items_rank) >= self.n)
            if break_condition1:
                break
        else:
            # all related users has been checked, and haven't meet the break condition
            # so just return the current items_rank
            print('Not enough users to meet k or items to meet n: {}, {}'.
                  format(n_similar_users, len(items_rank)))
        return items_rank

    def make_recommendation(self, user_id):
        super().make_recommendation(user_id)
        related_users = self.sim_matrix[user_id]
        if len(related_users) == 0:
            print('user {} didn\'t has any common item with other users')
            return -2
        related_users = sorted(related_users.items(),
                               key=lambda item: item[1],
                               reverse=True)  # return a list of tuples
        related_users_id = [x[0] for x in related_users]
        items_rank = self.rank_potential_items(user_id, related_users_id)
        if self.ensure_new and len(items_rank) == 0:
            print('All recommend items has already been bought by user {}.'.format(user_id))
            return -3
        return super().get_top_n_items(items_rank)

    def evaluate(self, test_data):
        return super().evaluate_recommendation(test_data)
