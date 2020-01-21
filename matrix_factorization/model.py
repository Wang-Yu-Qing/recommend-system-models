from keras.layers import Input, Embedding, Flatten, dot
import keras
import pandas as pd

"""
    data used can be downloaded from:
    https://www.kaggle.com/retailrocket/ecommerce-dataset
"""


class Matrix_factorization(object):
    def __init__(self, paras):
        self.paras = paras

    def construct(self):
        # 商品id嵌入网络结构
        item_input = Input(shape=[1], name='Items')
        item_embed = Embedding(input_dim=self.paras['item_embed_input_dim'],
                               output_dim=self.paras['item_embed_output_dim'],
                               input_length=1,
                               name='Items_embedding')(item_input)
        item_vec = Flatten(name='Items_flatten')(item_embed)
        # 用户id嵌入网络结构, 嵌入层不包含偏置参数，因此计算参数数量时不考虑偏置项，也不包含激活函数。
        visitor_input = Input(shape=[1], name='Vsitors')
        visitor_embed = Embedding(input_dim=self.paras['visitor_embed_input_dim'],  # noqa
                                  output_dim=self.paras['visitor_embed_output_dim'],  # noqa
                                  input_length=1,
                                  name='Visitors_embedding')(visitor_input)
        visitor_vec = Flatten(name='Visitors_flatten')(visitor_embed)
        # 商品嵌入结果向量与用户嵌入结果向量的dot product,
        # 这里通过向量点积合并两个向量。
        dot_product = dot([item_vec, visitor_vec],
                          axes=[1, 1], name='Dot_product')
        self.model = keras.Model([item_input, visitor_input], dot_product)
        self.model.compile('adam', 'mse')
        # 绘制模型结构
        keras.utils.plot_model(self.model,
                               to_file='matrix_factorization/model_struc/model.png',  # noqa
                               show_shapes=True, show_layer_names=True)
        # 显示模型结构概述
        self.model.summary()

    def train(self, train_data):
        self.model.fit([train_data['itemid'], train_data['visitorid']],
                       train_data['event'], epochs=50)

    def evaluate(self, test_data):
        score = self.model.evaluate([test_data['itemid'],
                                     test_data['visitorid']],
                                    test['event'])
        print('mse: {}'.format(score))
