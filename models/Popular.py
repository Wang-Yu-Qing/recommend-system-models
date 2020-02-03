from Base_model import Base_model


class Popular(Base_model):
    def __init__(self, n, data_type, ensure_new=True):
        super().__init__(n, "MostPopular", data_type, ensure_new=ensure_new)

    def fit(self, event_data, force_training):
        super().fit(event_data, force_training)
        # sort items by popularity, return a sorted list of tuples
        self.items = sorted(self.items.items(),
                            key=lambda item: len(item[1].covered_users),
                            reverse=True)

    def make_recommendation(self, user_id):
        user = self.users[user_id]
        history_items = user.covered_items
        items_rank = []
        for item_id, item in self.items:
            if item_id in history_items:
                continue
            items_rank.append(item_id)
            if len(items_rank) == self.n:
                break
        else:
            print("[{}] Not enough n untouched items for user {}".format(self.name, user_id))  # noqa
            print("[{}] Recommend {} items instead".format(self.name, len(items_rank)))  # noqa
        return items_rank

    def evaluate(self, test_data):
        super().evaluate_recommendation(test_data)
