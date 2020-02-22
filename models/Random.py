from base.Model import Model
import random


class Random(Model):
    def __init__(self, n, data_type, ensure_new=True):
        super().__init__(n, "random", data_type, ensure_new=ensure_new)

    def fit(self, event_data):
        super().fit(event_data)
        self.save()

    def make_recommendation(self, user_id):
        user = self.users[user_id]
        history_items = user.covered_items
        items_rank = set()
        items_pool = list(self.items.keys())
        while len(items_rank) < self.n and items_pool:
            # random choose an new item for this user
            rand_index = random.randint(0, len(items_pool)-1)
            item_id = items_pool.pop(rand_index)
            if item_id in history_items:
                continue
            items_rank.add(item_id)
        if not items_pool and (len(items_rank) < self.n):
            print("[{}] Not enough n untouched items for user {}".format(self.name, user_id))
            print("[{}] Recommend {} items instead".format(self.name, len(items_rank)))
        return items_rank

    def evaluate(self, test_data):
        return super().evaluate_recommendation(test_data)
