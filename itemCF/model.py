from userCF.model import UserCF
from userCF.user import User
from userCF.item import Item
import pandas as pd
import pickle


class ItemCF(UserCF):
    def __init__(self, all_id, k, n, ensure_new=True, IIF=False):
        super().__init__(all_id, k, n, ensure_new=ensure_new, IIF=IIF)

    def init_item_and_user_objects(self, event_data):
        return super().init_item_and_user_objects(event_data)

    def update_item_item_sim(self):
        pass

    def compute_item_item_sim_based_on_common_users(self):
        pass

    def standardize_sim_values(self):
        pass

    def build_item_item_sim_matrix(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def rank_potential_items(self, target_user_id):
        pass

    def get_top_n_items(self, items_rank):
        return super().get_top_n_items(items_rank)

    def normalize_sim_for_u(self):
        pass

    def make_recommendation(self):
        pass

    def compute_n_hit(self, user_id, real_items):
        return super().compute_n_hit(user_id, real_items)

    def evaluate(self):
        pass


def train_item_cf_model():
    pass


def evaluate_item_cf_model():
    pass
