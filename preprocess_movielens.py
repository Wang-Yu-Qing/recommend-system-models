import os
import pandas as pd
from datetime import datetime


def read_and_parse():
    df = []
    with open('data/MovieLens/ratings.dat', 'r') as f:
        line = f.readline()
        i = 1
        while line:
            values = line.split('::')
            # convert timestamp to string
            values[-1] = datetime.utcfromtimestamp(int(values[-1])).\
                strftime('%Y-%m-%d %H:%M:%S')
            i += 1
            line = f.readline()
            df.append(values)
    df = pd.DataFrame(df,
                      columns=['visitorid', 'itemid', 'rating', 'time'])
    # make it the same format as retailrocket
    df['event'] = 0  # 0 for all rows take into account
    # shuffle the data, make the model see most of the
    # users during the training procedure
    df = df.sample(frac=1)
    df.to_csv('data/MovieLens/ratings.csv', index=False)


if __name__ == '__main__':
    read_and_parse()
