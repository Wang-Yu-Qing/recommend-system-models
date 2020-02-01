from itertools import islice
import pandas as pd
import os
from glob import glob
from Recommend_model import Recommend_model


class Data_util:
    def __init__(self, data_type):
        data_types = [dir_name.split("/")[1] for dir_name in glob("data/*")]
        if data_type not in data_types:
            raise ValueError('Wrong data {} type provided, must be in {}'.
                             format(data_type, data_types))
        self.data_type = data_type

    @staticmethod
    def parse_line(row, sep):
        row = row.split(sep)
        # remove \n
        row[-1] = row[-1][:-1]
        row = [int(value) for value in row]
        return row

    def read_event_data(self, test_size=0.25):
        if self.data_type[-1] == 'K':
            sep = "\t"
        elif self.data_type[-1] == 'M':
            sep = "::"
        else:
            raise ValueError("[data_util] Invalid data type name.")
        with open(os.path.join("data",
                               self.data_type,
                               "ratings.dat"), 'r') as f:
            data = [self.parse_line(row, sep) for row in islice(f, None)]
        split = int((1-test_size)*len(data))
        data = pd.DataFrame(data, columns=['visitorid',
                                           'itemid',
                                           'rating',
                                           'timestamp'])
        data = data.sample(frac=1, random_state=100)
        train, test = data.iloc[:split, :], data.iloc[split:, :]
        return train, test

    def create_negative_samples_for_single_user(self, user, items_pop,
                                                negative_samples,
                                                neg_frac):
        """
            items pop must be sorted in decreasing order
        """
        n_created = 0
        n_pos = len(user.covered_items)
        n_neg = int(neg_frac * n_pos)
        for item_id, pop in items_pop:
            if item_id not in user.covered_items:
                negative_samples.append([user.id, item_id, 0])
                n_created += 1
            if n_created == n_neg:
                break
        else:
            print("""Not enough untouched items for user {} to create {} negative samples, create {} instead.""".format(user.id, n_neg, n_created))  # noqa

    def create_negative_samples(self, pos_samples, neg_frac):
        """
        create negative samples for each user
        if the item is not touched by the user
        and its popularity is high, then mark
        this item as user's negative sample
        """
        self.items, self.users = Recommend_model.init_item_and_user_objects(pos_samples)  # noqa
        # sort items by popularity
        item_pop = {}
        for item_id, item in self.items.items():
            item_pop[item_id] = len(item.covered_users)
        # return list for tuples
        items_pop = sorted(item_pop.items(),
                           key=lambda item: item[1],
                           reverse=True)
        negative_samples = []
        for user in self.users.values():
            self.create_negative_samples_for_single_user(user, items_pop,
                                                         negative_samples,
                                                         neg_frac)
        return pd.DataFrame(negative_samples,
                            columns=['visitorid', 'itemid', 'event'])


if __name__ == '__main__':
    Data_util('MovieLens_1M')
