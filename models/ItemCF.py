import os
import pandas as pd
import pickle
from base.Model import Model
from base.User import User
from base.Item import Item
from math import sqrt, log

class ItemCF(Model):
    def __init__(self, n, k, data_type, ensure_new=True, timestamp=False):
        super().__init__(n, "ItemCF", data_type, ensure_new=ensure_new)
        self.k = k
        self.name += "_k_{}".format(k)
        if timestamp:
            self.name += "_TimeContext"
        self.timestamp = timestamp

    def update_item_item_sim(self, item_A_info, item_B_info, user_frequency):
        item_A_id, item_A_time = item_A_info
        item_B_id, item_B_time = item_B_info
        try:
            row = self.sim_matrix[item_A_id]
        except KeyError:
            row = self.sim_matrix[item_A_id] = {}
        # compute score
        IUF = log(1+user_frequency)
        if self.timestamp:
            time_elapse = Model.time_elapse(item_A_time, item_B_time)
            score = 1*time_elapse/IUF
        else:
            score = 1/IUF
        # update score
        try:
            row[item_B_id] += score
        except KeyError:
            row[item_B_id] = score

    def compute_item_item_sim_based_on_common_users(self):
        for user in self.users.values():
            user_freq = len(user.covered_items)
            # update each items pair for this user
            items = list(user.covered_items.items())
            for i in range(len(items)-1):
                for j in range(i+1, len(items)):
                    item_A_info, item_B_info = items[i], items[j]
                    self.update_item_item_sim(item_A_info, item_B_info, user_freq)  # noqa
                    self.update_item_item_sim(item_B_info, item_A_info, user_freq)  # noqa

    def standardize_sim_values(self):
        for item_id_A, sim_dict in self.sim_matrix.items():
            for item_id_B, sim in sim_dict.items():
                item_pop_A = len(self.items[item_id_A].covered_users)
                item_pop_B = len(self.items[item_id_B].covered_users)
                sim /= sqrt(item_pop_A*item_pop_B)
                assert sim <= 1

    def fit(self, event_data):
        if super().fit(event_data):
            return
        self.sim_matrix = {}
        print("[{}] Building item-item similarity matrix, this may take some time...".format(self.name))  # noqa
        self.compute_item_item_sim_based_on_common_users()
        self.standardize_sim_values()
        print("[{}] Build done!".format(self.name))
        self.save()

    def rank_potential_items(self, user_id, all_k_sim_items):
        items_rank = {}
        history_items_id = all_k_sim_items.keys()
        user = self.users[user_id]
        for history_item_id, items_sim in all_k_sim_items.items():
            # get the timestamp when the last time this history item
            # been touched by the user
            t_history_item = user.covered_items[history_item_id]
            for item_id, sim in items_sim:
                if self.ensure_new and (item_id in history_items_id):
                    continue
                # compute score
                if self.timestamp:
                    # user's interest for this history item
                    t_now = 1146454548
                    time_elapse = Model.time_elapse(t_now, t_history_item)
                    score = time_elapse*sim
                else:
                    # if not consider time context, for history touched items
                    # user's interest is 1
                    score = 1*sim
                try:
                    items_rank[item_id] += score
                except KeyError:
                    items_rank[item_id] = score
        # assert len(items_rank) >= self.n
        return items_rank

    def normalize_k_items_sim(self, k_items):
        # for one single history item, normalize its most similar k
        # items' similarity values so that they sum to 1
        sum_value = sum([sim for item_id, sim in k_items])
        normalized_sim = []
        for item_id, sim in k_items:
            normalized_sim.append((item_id, sim/sum_value))
        return normalized_sim

    def normalize_sim(self, history_items):
        # keys are item_id of history items,
        # values are list of tuples of history item's k similar items
        # and their similarity
        for item_id, k_items in history_items.items():
            # k_items = self.normalize_k_items_sim(k_items)  # never do this!
            history_items[item_id] = self.normalize_k_items_sim(k_items)

    def make_recommendation(self, user_id):
        if not super().valid_user(user_id):
            return -1
        # for each item in the user history, find its k most similar items
        history_items_id = self.users[user_id].covered_items
        all_k_sim_items = {}
        # {history_item_A: {most_similar_item: sim1,
        #                   second_similar_item: sim2, ...},
        #  history_item_B: {...}}
        for item_id in history_items_id:
            # Because we need to iter through every history item,
            # for user with large amount of history items list,
            # this step can be SLOW.
            sim_dict = self.sim_matrix[item_id]
            # get this item's k most similar items
            # sorted return list of tuples
            sim_tuple = sorted(sim_dict.items(),
                               key=lambda item: item[1],
                               reverse=True)
            try:
                all_k_sim_items[item_id] = sim_tuple[:self.k]
            except IndexError:
                all_k_sim_items[item_id] = sim_tuple
        self.normalize_sim(all_k_sim_items)
        items_rank = self.rank_potential_items(user_id, all_k_sim_items)
        items_id = super().get_top_n_items(items_rank)
        return items_id

    def evaluate(self, test_data):
        return super().evaluate_recommendation(test_data)

    def save(self):
        super().save()
        sim_matrix = os.path.join('models/saved_models/sim_matrix_{}'.format(self.name + '.pickle'))
        with open(sim_matrix, 'wb') as f:
            f.write(pickle.dumps(self.sim_matrix))
        print("[{}] Model saved.".format(self.name))

    def load(self):
        super().load()
        sim_matrix = os.path.join('models/saved_models/sim_matrix_{}'.format(self.name + '.pickle'))
        with open(sim_matrix, 'rb') as f:
            self.sim_matrix = pickle.loads(f.read())
        print("[{}] Previous sim matrix found and loaded.".format(self.name))  # noqa
