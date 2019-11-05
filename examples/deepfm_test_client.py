# -*- coding: utf-8 -*-
"""
 @Time    : 2019-11-05 11:09
 @Author  : sishuyong 
 @Email   : shuyong1@staff.weibo.com
 @File    : deepfm_test_client.py
 @brief   : 
"""
from tensorflow.python.keras.models import save_model, load_model
from deepctr.layers import custom_objects

import keras
import os
import tensorflow as tf
from tensorflow.python.util import compat
from keras import backend as K


def export_savedmodel(model):
    '''
    传入keras model会自动保存为pb格式
    '''
    model_path = "model/"  # 模型保存的路径
    model_version = 0  # 模型保存的版本
    # 从网络的输入输出创建预测的签名
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': model.input}, outputs={'output': model.output})
    # 使用utf-8编码将 字节或Unicode 转换为字节
    export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))  # 将保存路径和版本号join
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)  # 生成"savedmodel"协议缓冲区并保存变量和模型
    builder.add_meta_graph_and_variables(  # 将当前元图添加到savedmodel并保存变量
        sess=K.get_session(),  # 返回一个 session 默认返回tf的sess,否则返回keras的sess,两者都没有将创建一个全新的sess返回
        tags=[tf.saved_model.tag_constants.SERVING],  # 导出模型tag为SERVING(其他可选TRAINING,EVAL,GPU,TPU)
        clear_devices=True,  # 清除设备信息
        signature_def_map={  # 签名定义映射
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  # 默认服务签名定义密钥
                model_signature  # 网络的输入输出策创建预测的签名
        })
    builder.save()  # 将"savedmodel"协议缓冲区写入磁盘.
    print("save model pb success ...")


def load_and_test():
    model = load_model("./keras_model/deepfm.h5", custom_objects)
    print(model)
    return model
    # # 代码:
    # import tensorflow as tf
    #
    # # 导出路径包含模型的名称和版本
    # tf.keras.backend.set_learning_phase(0)
    # # model = tf.keras.models.load_model('./inception.h5')  # 需要加载的模型路径
    # export_path = './tf_models/deepfm'  # 将要导出模型的路径
    #
    # # 获取Keras会话和保存模型
    # # 签名的定义是定义的输入和输出张量
    # with tf.keras.backend.get_session() as sess:
    #     tf.saved_model.simple_save(
    #         sess,
    #         export_path,
    #         model.input,
    #         model.output
    #     )


if __name__ == "__main__":
    model = load_and_test()

    export_savedmodel(model)  # 将模型传入保存模型的方法内,模型保存成功.
