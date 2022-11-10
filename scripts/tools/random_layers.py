import numpy as np

np.random.seed(20210808)


def random_layer(layer_name):
    if layer_name == 'Conv2D':
        return {'activation': np.random.choice(
            ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
             'exponential', 'linear']),
            'kernel_initializer': np.random.choice(
                ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                 'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                 'he_normal', 'lecun_normal'])}
    elif layer_name == 'DepthwiseConv2D':
        return {'activation': np.random.choice(
            ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
             'exponential', 'linear']),
            'kernel_initializer': np.random.choice(
                ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                 'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                 'he_normal', 'lecun_normal'])}
    elif layer_name == 'SeparableConv2D':
        return {'activation': np.random.choice(
            ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
             'exponential', 'linear']),
            'kernel_initializer': np.random.choice(
                ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                 'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                 'he_normal', 'lecun_normal'])}
    elif layer_name == 'Activation':
        return {'activation': np.random.choice(
            ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
             'exponential', 'linear'])}
    elif layer_name == 'ReLU':
        return {'max_value': np.random.random(1)[0],
                'negative_slope': np.random.random(1)[0],
                'threshold': np.random.random(1)[0]}
    elif layer_name == 'LeakyReLU':
        return {'alpha': np.random.random(1)[0]}
    elif layer_name == 'PReLU':
        return {'alpha_initializer': np.random.choice(
            ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling',
             'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal'])}
    elif layer_name == 'ELU':
        return {'alpha': np.random.random(1)[0]}
    elif layer_name == 'ThresholdedReLU':
        return {'theta': np.random.random(1)[0]}
    elif layer_name == 'BatchNormalization':
        return {'momentum': np.random.random(1)[0],
                'epsilon': np.random.random(1)[0]}
    elif layer_name == 'SimpleRNN':
        return {'activation': np.random.choice(
            ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
             'hard_sigmoid', 'exponential', 'linear']),
            'dropout': np.random.random(1)[0],
            'recurrent_dropout': np.random.random(1)[0]}
    else:
        assert f"OP {layer_name} is not support yet!"


def is_layer_in_random_layer_list(layer):
    import keras
    layer_list = [keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.SeparableConv2D,
                  keras.layers.Activation, keras.layers.ReLU, keras.layers.LeakyReLU,
                  keras.layers.PReLU, keras.layers.ELU, keras.layers.ThresholdedReLU,
                  keras.layers.BatchNormalization, keras.layers.SimpleRNN
                  ]
    # print(white_list)
    for l in layer_list:
        if isinstance(layer, l):
            return True
    return False


def random_para_layer(layer, changed_dict):
    new_config = layer.get_config()
    for para in changed_dict:
        new_config[para] = changed_dict[para]
        # print(para, v)
    new_config['name'] = new_config['name']
    # print(new_config['name'])
    new_layer = layer.__class__.from_config(new_config)
    return new_layer
