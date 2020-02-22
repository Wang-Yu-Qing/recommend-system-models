from itertools import islice
import pandas as pd
import os
from glob import glob
from base.Model import Model


class Data_util:
    data_paths = {
        "MovieLens_1M": "data/MovieLens_1M/ratings.dat",
        "MovieLens_100K": "data/MovieLens_100K/ratings.dat",
        "Hetrec-2k": "data/Hetrec-2k/user_taggedartists-timestamps.dat"
    }

    def __init__(self, data_type):
        # get available data folder names
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
        return row

    @staticmethod
    def sort_user_actions(event_data, test_size):
        train, test = pd.DataFrame(), pd.DataFrame()
        users_id = pd.unique(event_data['visitorid'])
        for user_id in users_id:
            user_actions = event_data.loc[event_data['visitorid']== user_id, :]
            user_actions = user_actions.sort_values(by=["timestamp"])
            split = int(test_size*len(user_actions))
            test = pd.concat([test, user_actions.iloc[:split]],
                             ignore_index=True)
            train = pd.concat([train, user_actions.iloc[split:]],
                              ignore_index=True)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def read_event_data(self, test_size=0.25):
        skip_first_row = False
        if self.data_type[-1] == 'K':
            sep = "\t"
        elif self.data_type == "Hetrec-2k":
            sep = "\t"
            skip_first_row = True
        elif self.data_type[-1] == 'M':
            sep = "::"
        else:
            raise ValueError("[data_util] Invalid data type name.")
        with open(self.data_paths[self.data_type], 'r') as f:
            if skip_first_row:
                f.readline()  # make the generator one step forward
            data = [self.parse_line(row, sep) for row in islice(f, None)]
        if self.data_type == "Hetrec-2k":
            cols = ["visitorid", "itemid", "tagid", "timestamp"]
        else:
            cols = ["visitorid", "itemid", "rating", "timestamp"]
        data = pd.DataFrame(data, columns=cols)
        # data type convert
        data["timestamp"] = data["timestamp"].apply(lambda x: int(x))
        data["visitorid"] = data["visitorid"].apply(lambda x: int(x))
        data["itemid"] = data["itemid"].apply(lambda x: int(x))
        train, test = self.sort_user_actions(data, test_size)
        return train, test

    @staticmethod
    def join_movie_lens_event_data(event_data, users_info, items_info):
        # convert data type before join
        # false positive detection may occur
        # https://stackoverflow.com/questions/23688307/settingwithcopywarning-even-when-using-loc
        # join with user info
        event_data = event_data.merge(users_info, how="left", on="visitorid")
        # join with item info
        event_data = event_data.merge(items_info, how="left", on="itemid")
        return event_data

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
        self.items, self.users = Model.init_item_and_user_objects(pos_samples)  # noqa
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

    def build_samples(self, neg_frac, train_event_data):
        """ return all samples
        """
        train_event_data["event"] = 1
        pos_samples = train_event_data
        neg_samples = self.create_negative_samples(pos_samples, neg_frac)
        # timestamp, rating for negative samples are NA
        samples = pd.concat([neg_samples, pos_samples], ignore_index=True, sort=False)
        samples = samples.sample(frac=1, random_state=100).reset_index(drop=True)  # noqa
        return samples

    @staticmethod
    def split_samples(samples, test_size):
        split = int(len(samples)*test_size)
        test, train = samples.iloc[:split, :], samples.iloc[split:, :]
        return train, test


if __name__ == '__main__':
    Data_util('MovieLens_1M')
