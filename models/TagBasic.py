from base.Model import Model
from math import log
import pandas as pd

class TagBasic(Model):
    def __init__(self, n, k, data_type, ensure_new=True):
        super().__init__(n, "TagBasic", data_type, ensure_new=ensure_new)
        self.k = k
        self.name += "_{}".format(k)

    def fit(self, train_data):
        super().fit(train_data, tag=True)
        self.save()

    def find_k_most_used_tag(self, user):
        tags_count_sorted = sorted(user.tags_count.items(),
                                   key=lambda item: item[1],
                                   reverse=True)
        return tags_count_sorted[:self.k]  # lest than k will still return

    def rank_potential_items(self, user, k_tags):
        items_rank = {}
        for tag in k_tags:
            tag_pop = sum(tag.items_count.values())
            tag_freq_user = user.tags_count[tag.id]
            for item_id, tag_freq_item in tag.items_count.items():
                # check if this item has been touched by the user
                if item_id in user.covered_items:
                    continue
                item_pop = len(self.items[item_id].covered_users)
                # compute this item's score
                score = (tag_freq_user/log(1+tag_pop)) * (tag_freq_item/log(1+item_pop))
                # sum to corresponding item's previous score
                try:
                    items_rank[item_id] += score
                except KeyError:
                    items_rank[item_id] = score
        return items_rank

    def make_recommendation(self, user_id):
        try:
            user = self.users[user_id]
        except KeyError:
            print("[{}] User {} not seen in training set".format(user_id))
        # find user's k most used tag
        most_used_tag = self.find_k_most_used_tag(user)
        if len(most_used_tag) < self.k:
            print("[{}] User {} used tags number less than k".format(self.name, user_id))  # noqa
        # for every tag in k, find all items tagged this tag
        k_tags = []
        for tag_id in [x[0] for x in most_used_tag]:
            k_tags.append(self.tags[tag_id])
        items_rank = self.rank_potential_items(user, k_tags)
        if len(items_rank) >= self.n:
            return self.get_top_n_items(items_rank)
        else:
            return -1

    def evaluate(self, test_data):
        return super().evaluate_recommendation(test_data)
