# 数据描述： https://www.kaggle.com/retailrocket/ecommerce-dataset/data
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_ids(df):
    """
        嵌入层会根据输入的数值进行权重矩阵的look up，通过索引选出对应的权重行，
        这与one-hot encoding的效果是一致的，嵌入层的行数，即input_dim参数，
        即可能存在的最大类别数，必须大于等于输入的可能数值，否则索引越界，引发异常。
        若input_dim取得远大于训练及测试数据中可能的类别最大值，或类别取值中间不连续、不从0开始，
        则embedding层中有一些权重行在训练及测试过程中将不被选中，即权重初始化后保持不变，
        不影响训练结果，只是在内存中有一些浪费，但若新的预测数据中包含训练测试集中不包含，但又小于
        input_dim的encode值，则其embedding结果是最初的随机初始化值，就会造成精度问题。
        https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer
        https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_11_03_embedding.ipynb
        因此只需要将离散变量的值encode成整数即可，不必变为one-hot，以
        节省内存空间。但需要保证可能取到的整数最大值须小于embedding层的input_dim。
        https://stackoverflow.com/questions/56227671/how-can-i-one-hot-encode-a-list-of-strings-with-keras

    Args:
        df ([type]): [description]
    """
    # 尽量使id从0开始，连续到最大值
    encoder = LabelEncoder()
    df['visitorid'] = encoder.fit_transform(df['visitorid'])
    df['itemid'] = encoder.fit_transform(df['itemid'])


def preprocess_event_data(df):
    # 将用户行为转化为数值格式
    df['event'].replace({'view': 1, 'addtocart': 2, 'transaction': 3},
                        inplace=True)
    # 删除无用列
    del df['transactionid'], df['timestamp']
    # 保证所有可能的用户及物品id都被encode上
    encode_ids(df)
    print(len(pd.unique(df['visitorid'])), max(df['visitorid']), min(df['visitorid']))
    print(len(pd.unique(df['itemid'])), max(df['itemid']), min(df['itemid']))
    # 打乱数据顺序
    df = df.sample(frac=1).reset_index(drop=True)
    # 分割训练、测试、验证集
    max_index = len(df)-1
    p1 = int(max_index*0.7)
    train, test = df.iloc[:p1, :], df.iloc[p1:, :]
    return train, test


if __name__ == '__main__':
    events = pd.read_csv('data/Retailrocket/events.csv')
    train, test = preprocess_event_data(events)
    events.to_csv('data/Retailrocket/events_id_encoded.csv', index=False)
