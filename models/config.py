# @Time    : 2019/4/17 21:17
# @Author  : shakespere
# @FileName: config.py
"""模型参数设置，训练参数设置"""


class LSTMConfig:
    """LSTM模型参数"""
    emb_size = 300
    hidden_size = 128

class ATTConfig:
    """ATT模型参数"""
    emb_size = 300
    hidden_size = 128

class RCNNConfig:
    """ATT模型参数"""
    emb_size = 300
    hidden_size = 128

class LSTMATTConfig:
    """ATT模型参数"""
    emb_size = 300
    hidden_size = 128

class CNNConfig:
    """CNN模型参数"""
    emb_size = 300
    num_filters = 2  # 每种窗口对应的输出channel个的个数
    window_sizes = [3, 4, 5]  # 窗口大小

class ATTTrainingConfig:
    """设置LSTM模型训练的参数"""
    batch_size = 128
    learning_rate = 0.003
    epoches = 8
    print_step = 100

    # ReduceLROnPlateau参数
    factor = 0.5
    patience = 1
    verbose = True

class RCNNTrainingConfig:
    """设置LSTM模型训练的参数"""
    batch_size = 128
    learning_rate = 0.003
    epoches = 8
    print_step = 100

    # ReduceLROnPlateau参数
    factor = 0.5
    patience = 1
    verbose = True

class LSTMATTTrainingConfig:
    """设置LSTM模型训练的参数"""
    batch_size = 128
    learning_rate = 0.003
    epoches = 8
    print_step = 100

    # ReduceLROnPlateau参数
    factor = 0.5
    patience = 1
    verbose = True

class LSTMTrainingConfig:
    """设置LSTM模型训练的参数"""
    learning_rate = 0.003
    epoches = 8
    print_step = 100

    # ReduceLROnPlateau参数
    factor = 0.5
    patience = 1
    verbose = True


class CNNTrainingConfig:
    """设置CNN模型训练的参数"""
    learning_rate = 0.0015
    epoches = 8
    print_step = 64

    # ReduceLROnPlateau参数
    factor = 0.3
    patience = 1
    verbose = True


class LRConfig:
    """LogisticRegression参数"""
    learning_rate = 0.0001
    epoches = 15