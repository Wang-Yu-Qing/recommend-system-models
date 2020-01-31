from itertools import islice
import pandas as pd
import os
from glob import glob


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


if __name__ == '__main__':
    Data_util('MovieLens_1M')
