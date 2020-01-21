import os
import pandas as pd


def read_and_parse():
    df = []
    with open('data/MovieLens/ratings.dat', 'r') as f:
        line = f.readline()
        while line:
            values = line.split('::')
            df.append(values)
    df = pd.DataFrame(df, columns=['userid', 'movieid', 'rating', 'timestamp'])
    df.to_csv('data/MovieLens/ratings.csv', index=False)


if __name__ == '__main__':
    read_and_parse()
