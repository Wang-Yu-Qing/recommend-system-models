from userCF.model import UserCF
from userCF.user import User
from userCF.item import Item
from math import sqrt, log
import os
import csv
import pandas as pd
import pickle


class ItemCF(UserCF):
    def __init__(self, all_id, k, n, ensure_new=True, IIF=False):
        super().__init__(all_id, k, n, ensure_new=ensure_new, IIF=IIF)

    def init_item_and_user_objects(self, event_data):
        super().init_item_and_user_objects(event_data)

    def update_item_item_sim(self, item_A_id, item_B_id, user_frequency):
        item_A = self.sim_matrix[item_A_id]
        if self.IIF:
            try:
                item_A[item_B_id] += 1/log(1+user_frequency)
            except KeyError:
                item_A[item_B_id] = 1/log(1+user_frequency)
        else:
            try:
                item_A[item_B_id] += 1
            except KeyError:
                item_A[item_B_id] = 1

    def compute_item_item_sim_based_on_common_users(self):
        for user in self.users.values():
            user_freq = len(user.covered_items)
            # update each items pair for this user
            items = list(user.covered_items)
            for i in range(len(items)-1):
                for j in range(i+1, len(items)):
                    item_A_id, item_B_id = items[i], items[j]
                    self.update_item_item_sim(item_A_id, item_B_id, user_freq)
                    self.update_item_item_sim(item_B_id, item_A_id, user_freq)

    def standardize_sim_values(self):
        for item_id, sim_dict in self.sim_matrix.items():
            for another_item_id, sim in sim_dict.items():
                item_pop = len(self.items[item_id].covered_users)
                another_item_pop = len(self.items[another_item_id].
                                       covered_users)
                sim /= sqrt(item_pop*another_item_pop)
                assert sim <= 1

    def build_item_item_sim_matrix(self, event_data):
        bool_index = event_data['event'] != 1
        event_data = event_data.loc[bool_index, :]
        self.init_item_and_user_objects(event_data)
        self.compute_item_item_sim_based_on_common_users()
        self.standardize_sim_values()

    def save(self, dir):
        """
            save the whole model
        """
        if self.IIF:
            file_name = 'item_sim_matrix_IIF.pickle'
        else:
            file_name = 'item_sim_matrix.pickle'
        with open(os.path.join(dir, file_name), 'wb') as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load(dir, IIF):
        """
            load the whole model
        """
        if IIF:
            file_name = 'item_sim_matrix_IIF.pickle'
        else:
            file_name = 'item_sim_matrix.pickle'
        with open(os.path.join(dir, file_name), 'rb') as f:
            model = pickle.loads(f.read())
        return model

    def rank_potential_items(self, history_items_id, k_sim_items_for_each):
        items_rank = {}
        for item_sim in k_sim_items_for_each.values():
            for item_id, sim in item_sim:
                if self.ensure_new and (item_id in history_items_id):
                    continue
                try:
                    items_rank[item_id] += sim
                except KeyError:
                    items_rank[item_id] = sim
        return items_rank

    def get_top_n_items(self, items_rank):
        return super().get_top_n_items(items_rank)

    def normalize_sim_for_item_u(self, k_sim_items_for_each):
        # keys are item_id of history items, values are tuples
        for item_id, k_items in k_sim_items_for_each.items():
            # normalize so that sum as 1
            sum_value = sum([item_sim[1] for item_sim in k_items])
            normalized_sim = []
            for item, sim in k_items:
                normalized_sim.append((item, sim/sum_value))
            k_sim_items_for_each[item_id] = normalized_sim

    def make_recommendation(self, user_id):
        # get user's covered items in training set
        try:
            history_items_id = self.users[user_id].covered_items
        except KeyError:
            print('User not shown in the training set, cannot make recommendation')
            return -1
        # for each item in the user history, find its k most similar items
        k_sim_items_for_each = {}
        # {history_item_A: {most_similar_item: sim1, second_similar_item: sim2, ...},
        #  history_item_B: {...}}
        for item_id in history_items_id:
            # Because we need to iter through every history item, for user with large amount
            # of history items list, this step can be SLOW.
            # get this item's k most similar items
            sim_dict = self.sim_matrix[item_id]
            # sorted return list of tuples
            sim_tuple = sorted(sim_dict.items(),
                               key=lambda item: item[1],
                               reverse=True)
            try:
                k_sim_items_for_each[item_id] = sim_tuple[:self.k]
            except IndexError:
                k_sim_items_for_each[item_id] = sim_tuple
        self.normalize_sim_for_item_u(k_sim_items_for_each)
        items_rank = self.rank_potential_items(history_items_id,
                                               k_sim_items_for_each)
        items_id = self.get_top_n_items(items_rank)
        return items_id

    def compute_n_hit(self, reco_items_id, real_items_id):
        return super().compute_n_hit(reco_items_id, real_items_id)

    def evaluate(self, test_data):
        return super().evaluate(test_data)
    # def evaluate(self, test_data):
    #     users_id = pd.unique(test_data['visitorid'])
    #     recall, precision = 0, 0
    #     covered_items_id = set()
    #     for user_id in users_id:
    #         boolIndex = test_data['visitorid'] == user_id
    #         real_items = pd.unique(test_data.loc[boolIndex, 'itemid'])
    #         recommended_items_id = self.make_recommendation(user_id)
    #         n_hit = self.compute_n_hit(recommended_items_id, real_items)
    #         recall_user = n_hit/len(real_items)
    #         precision_user = n_hit/len(recommended_items_id)
    #         recall += recall_user
    #         precision += precision_user
    #         covered_items_id.update(recommended_items_id)
    #         print('user {} done with {} history items'.format(user_id,
    #                                                           len(self.users[user_id].covered_items)))
    #     n_users = len(users_id)
    #     recall /= n_users
    #     precision /= n_users
    #     coverage = len(covered_items_id)/len(self.items)
    #     return {'recall': recall, 'precision': precision, 'coverage': coverage}


def train_item_cf_model(portion, event_data_path, model_save_dir, IIF):
    event_data = pd.read_csv(event_data_path)
    # use small set for limited memory
    event_data = event_data.iloc[:int(len(event_data)*portion), :]
    split = int(len(event_data)*0.8)
    train = event_data.iloc[:split, :]
    items_id = pd.unique(train['itemid'])
    model = ItemCF(items_id, 0, 0, ensure_new=None, IIF=IIF)
    model.build_item_item_sim_matrix(train)
    model.save(model_save_dir)


def evaluate_item_cf_model(portion, event_data_path, model_save_dir,
                           k, n, ensure_new, IIF):
    # call static method without init class object
    model = ItemCF.load(model_save_dir, IIF)
    model.k, model.n = k, n
    model.ensure_new = ensure_new
    event_data = pd.read_csv(event_data_path)
    event_data = event_data.iloc[:int(len(event_data)*portion), :]
    bool_index = event_data['event'] != 1
    event_data = event_data.loc[bool_index, :]
    split = int(len(event_data)*0.8)
    test = event_data.iloc[split:, :]
    result = model.evaluate(test)
    # write result to file
    data_type = event_data_path.split('/')[1]
    with open('evaluation_results/itemCF-{}.csv'.format(data_type),
              'a', newline='') as f:
        cols = ['k', 'n', 'recall', 'precision',
                'coverage', 'ensure new', 'IIF']  # the order of the row
        writer = csv.DictWriter(f, delimiter=',', fieldnames=cols)
        writer.writerow({'k': k, 'n': n,
                         'recall': result['recall'],
                         'precision': result['precision'],
                         'coverage': result['coverage'],
                         'ensure new': ensure_new,
                         'IIF': IIF
                         })
