from base.Model import Model


class Popular(Model):
    def __init__(self, n, data_type, ensure_new=True):
        super().__init__(n, "MostPopular", data_type, ensure_new=ensure_new)

    def fit(self, event_data):
        super().fit(event_data)
        # sort items by popularity, return a sorted list of tuples
        self.items = sorted(self.items.items(),
                            key=lambda item: len(item[1].covered_users),
                            reverse=True)
        self.items = dict(self.items)
        self.save()

    def make_recommendation(self, user_id):
        user = self.users[user_id]
        history_items = user.covered_items
        items_rank = set()
        for item_id, item in self.items.items():
            if item_id in history_items:
                continue
            items_rank.add(item_id)
            if len(items_rank) == self.n:
                break
        else:
            print("[{}] Not enough n untouched items for user {}".format(self.name, user_id))  # noqa
            print("[{}] Recommend {} items instead".format(self.name, len(items_rank)))  # noqa
        return items_rank

    def evaluate(self, test_data):
        return super().evaluate_recommendation(test_data)
