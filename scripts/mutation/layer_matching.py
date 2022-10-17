import os
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class LayerMatching:
    concat_size_limit = 1e4

    def __init__(self):
        self.layers = {}
        self.constraints = {}

        self.layers['flatten'] = LayerMatching.flatten
        self.constraints['flatten'] = LayerMatching.flatten_constraints

        self.layer_concats = {}
        self.input_legal = {}
        self.layer_concats['flatten'] = LayerMatching.flatten_dense
        self.input_legal['flatten'] = LayerMatching.flatten_dense_input_legal
        self.layer_concats['repeat_vector'] = LayerMatching.repeat_vector_dense
        self.input_legal['repeat_vector'] = LayerMatching.repeat_vector_dense_input_legal
        self.layer_concats['cropping1d'] = LayerMatching.cropping1d_dense
        self.input_legal['cropping1d'] = LayerMatching.cropping1d_dense_input_legal
        self.layer_concats['cropping2d'] = LayerMatching.cropping2d_dense
        self.input_legal['cropping2d'] = LayerMatching.cropping2d_dense_input_legal
        self.layer_concats['cropping3d'] = LayerMatching.cropping3d_dense
        self.input_legal['cropping3d'] = LayerMatching.cropping3d_dense_input_legal
        self.layer_concats['upsampling_1d'] = LayerMatching.upsampling_1d_dense
        self.input_legal['upsampling_1d'] = LayerMatching.upsampling_1d_dense_input_legal
        self.layer_concats['upsampling_2d'] = LayerMatching.upsampling_2d_dense
        self.input_legal['upsampling_2d'] = LayerMatching.upsampling_2d_dense_input_legal
        self.layer_concats['upsampling_3d'] = LayerMatching.upsampling_3d_dense
        self.input_legal['upsampling_3d'] = LayerMatching.upsampling_3d_dense_input_legal
        self.layer_concats['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv
        self.input_legal['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv_input_legal
        self.layer_concats['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv
        self.input_legal['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv_input_legal
        self.layer_concats['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv
        self.input_legal['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv_input_legal
        self.layer_concats['global_max_pooling_1d'] = LayerMatching.global_max_pooling_1d_dense
        self.input_legal['global_max_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_average_pooling_1d'] = LayerMatching.global_average_pooling_1d_dense
        self.input_legal['global_average_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_max_pooling_2d'] = LayerMatching.global_max_pooling_2d_dense
        self.input_legal['global_max_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        self.layer_concats['global_average_pooling_2d'] = LayerMatching.global_average_pooling_2d_dense
        self.input_legal['global_average_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        self.layer_concats['global_max_pooling_3d'] = LayerMatching.global_max_pooling_3d_dense
        self.input_legal['global_max_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        self.layer_concats['global_average_pooling_3d'] = LayerMatching.global_average_pooling_3d_dense
        self.input_legal['global_average_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        self.layer_concats['simple_rnn'] = LayerMatching.simple_rnn_dense
        self.input_legal['simple_rnn'] = LayerMatching.simple_rnn_dense_input_legal
        self.layer_concats['gru'] = LayerMatching.gru_dense
        self.input_legal['gru'] = LayerMatching.gru_dense_input_legal
        self.layer_concats['lstm'] = LayerMatching.lstm_dense
        self.input_legal['lstm'] = LayerMatching.lstm_dense_input_legal
        self.layer_concats['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense
        self.input_legal['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense_input_legal
        # JTJ在下面扩展了MLA
        self.layer_concats['separable_conv_1d'] = LayerMatching.separable_conv_1d_dense
        self.input_legal['separable_conv_1d'] = LayerMatching.separable_conv_1d_dense_input_legal
        self.layer_concats['separable_conv_2d'] = LayerMatching.separable_conv_2d_dense
        self.input_legal['separable_conv_2d'] = LayerMatching.separable_conv_2d_dense_input_legal
        self.layer_concats['depthwise_conv_2d'] = LayerMatching.depthwise_conv_2d_dense
        self.input_legal['depthwise_conv_2d'] = LayerMatching.depthwise_conv_2d_dense_input_legal
        self.layer_concats['conv_2d_transpose'] = LayerMatching.conv_2d_transpose_dense
        self.input_legal['conv_2d_transpose'] = LayerMatching.conv_2d_transpose_dense_input_legal
        self.layer_concats['conv_3d_transpose'] = LayerMatching.conv_3d_transpose_dense
        self.input_legal['conv_3d_transpose'] = LayerMatching.conv_3d_transpose_dense_input_legal
        self.layer_concats['max_pooling_1d'] = LayerMatching.max_pooling_1d_dense
        self.input_legal['max_pooling_1d'] = LayerMatching.pooling_1d_dense_input_legal
        self.layer_concats['average_pooling_1d'] = LayerMatching.average_pooling_1d_dense
        self.input_legal['average_pooling_1d'] = LayerMatching.pooling_1d_dense_input_legal
        self.layer_concats['max_pooling_2d'] = LayerMatching.max_pooling_2d_dense
        self.input_legal['max_pooling_2d'] = LayerMatching.pooling_2d_dense_input_legal
        self.layer_concats['average_pooling_2d'] = LayerMatching.average_pooling_2d_dense
        self.input_legal['average_pooling_2d'] = LayerMatching.pooling_2d_dense_input_legal
        self.layer_concats['max_pooling_3d'] = LayerMatching.max_pooling_3d_dense
        self.input_legal['max_pooling_3d'] = LayerMatching.pooling_3d_dense_input_legal
        self.layer_concats['average_pooling_3d'] = LayerMatching.average_pooling_3d_dense
        self.input_legal['average_pooling_3d'] = LayerMatching.pooling_3d_dense_input_legal
        self.layer_concats['batch_normalization'] = LayerMatching.batch_normalization_dense
        self.input_legal['batch_normalization'] = LayerMatching.batch_normalization_dense_input_legal
        self.layer_concats['leaky_relu'] = LayerMatching.leaky_relu_dense
        self.input_legal['leaky_relu'] = LayerMatching.leaky_relu_dense_input_legal
        self.layer_concats['prelu'] = LayerMatching.prelu_dense
        self.input_legal['prelu'] = LayerMatching.prelu_dense_input_legal
        self.layer_concats['elu'] = LayerMatching.elu_dense
        self.input_legal['elu'] = LayerMatching.elu_dense_input_legal
        self.layer_concats['thresholded_relu'] = LayerMatching.thresholded_relu_dense
        self.input_legal['thresholded_relu'] = LayerMatching.thresholded_relu_dense_input_legal
        self.layer_concats['softmax'] = LayerMatching.softmax_dense
        self.input_legal['softmax'] = LayerMatching.softmax_dense_input_legal
        self.layer_concats['relu'] = LayerMatching.relu_dense
        self.input_legal['relu'] = LayerMatching.relu_dense_input_legal
        self.layer_concats['dropout'] = LayerMatching.dropout_dense
        self.input_legal['dropout'] = LayerMatching.dropout_dense_input_legal
        self.layer_concats['timeDistributed'] = LayerMatching.timeDistributed_dense
        self.input_legal['timeDistributed'] = LayerMatching.timeDistributed_dense_input_legal
        self.layer_concats['activity_normalization'] = LayerMatching.activity_normalization_dense
        self.input_legal['activity_normalization'] = LayerMatching.activity_normalization_dense_input_legal
        self.layer_concats['locally_Connected1D'] = LayerMatching.locally_Connected1D
        self.input_legal['locally_Connected1D'] = LayerMatching.locally_Connected1D_input_legal
        self.layer_concats['locally_Connected2D'] = LayerMatching.locally_Connected2D
        self.input_legal['locally_Connected2D'] = LayerMatching.locally_Connected2D_input_legal
        # JTJ在这里结束了扩展

    # JTJ添加了Reshape、Embedding块
    def reshape_block(input_shape, output_shape):
        import keras
        layer_concat = []
        if (len(input_shape) > 2):
            layer_concat.append(keras.layers.Flatten())
        units = 1
        for i in range(len(output_shape)):
            if i == 0:
                continue
            units *= output_shape[i]
        layer_concat.append(keras.layers.Dense(units))
        layer_concat.append(keras.layers.Reshape(output_shape[1:]))
        return layer_concat


    def reshape_block_input_legal(input_shape):
        units = 1
        for i in range(len(input_shape)):
            if i == 0:
                continue
            units *= input_shape[i]
        return units <= LayerMatching.concat_size_limit

    def embedding_dense(input_shape):
        # input_shape = input_shape.as_list()
        import keras
        layer = keras.layers.Embedding(1000, input_shape[1])
        layer.name += '_insert'
        return layer

    def embedding_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None

    # JTJ结束了Reshape块、Embedding块的添加

    @staticmethod
    def flatten(input_shape):
        import keras
        return keras.layers.Flatten()

    @staticmethod
    def flatten_constraints(input_shape):
        input_shape = input_shape.as_list()
        input_shape_len = len(input_shape)
        constraints = []
        if input_shape_len < 2:
            return None
        constraints = []
        dim_size = 1
        for i in range(input_shape_len):
            if i == 0:
                continue
            constraints.append('= input_{} {}'.format(i, input_shape[i]))
            dim_size *= input_shape[i]
        constraint_str = '= output_{} {}'.format(1, dim_size)
        constraints.append(constraint_str)
        return constraints

    # --------------------------------------------

    @staticmethod
    def flatten_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Flatten())
        units = 1
        for i in range(len(input_shape)):
            if i == 0:
                continue
            units *= input_shape[i]
        layer_concat.append(keras.layers.Dense(units))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def flatten_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        is_legal = len(input_shape) > 3 and input_shape[0] is None
        concat_size = 1
        for i, dim in enumerate(input_shape):
            if i == 0:
                continue
            is_legal = is_legal and dim is not None
            if dim is not None:
                concat_size *= dim
        return is_legal and concat_size <= LayerMatching.concat_size_limit

    @staticmethod
    def repeat_vector_dense(input_shape):
        n = 3
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.RepeatVector(n))
        layer_concat.append(keras.layers.Reshape((input_shape[1] * n,)))
        layer_concat.append(keras.layers.Dense(input_shape[1]))
        return layer_concat

    @staticmethod
    def repeat_vector_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[1] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping1D(cropping=(1, 1)))
        layer_concat.append(keras.layers.Dense(input_shape[1]))
        return layer_concat

    @staticmethod
    def cropping1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping2D(cropping=((1, 1), (1, 1))))
        layer_concat.append(keras.layers.Reshape(((input_shape[1] - 2) * (input_shape[2] - 2) * input_shape[3],)))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def cropping2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1))))
        layer_concat.append(keras.layers.Reshape(
            ((input_shape[1] - 2) * (input_shape[2] - 2) * (input_shape[3] - 2) * input_shape[4],)))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def cropping3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None and input_shape[3] > 2 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling1D(size=2))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        return layer_concat

    @staticmethod
    def upsampling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling2D(size=(2, 2)))
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def upsampling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling3D(size=(2, 2, 2)))
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def upsampling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_1d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding1D(padding=1))
        layer_concat.append(keras.layers.Conv1D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_1d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_2d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding2D(padding=(1, 1)))
        layer_concat.append(keras.layers.Conv2D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_2d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_3d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
        layer_concat.append(keras.layers.Conv3D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_3d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def simple_rnn_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.SimpleRNN(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def simple_rnn_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def gru_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GRU(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def gru_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def lstm_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.LSTM(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def lstm_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def conv_lstm_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same',
                                                    return_sequences=True))
        return layer_concat

    @staticmethod
    def conv_lstm_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] > 3 \
               and input_shape[3] is not None and input_shape[3] > 3 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    # JTJ在这里开始了添加
    @staticmethod
    def separable_conv_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.SeparableConv1D(input_shape[-1], 3, strides=1, padding='same'))
        return layer_concat

    @staticmethod
    def separable_conv_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3

    @staticmethod
    def separable_conv_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.SeparableConv2D(input_shape[-1], 3, strides=(1, 1), padding='same'))
        return layer_concat

    @staticmethod
    def separable_conv_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def depthwise_conv_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.DepthwiseConv2D(3, strides=(1, 1), padding='same'))
        return layer_concat

    @staticmethod
    def depthwise_conv_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_2d_transpose_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Conv2DTranspose(input_shape[-1], 3, strides=(1, 1), padding='same'))
        return layer_concat

    @staticmethod
    def conv_2d_transpose_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3

    @staticmethod
    def conv_3d_transpose_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Conv3DTranspose(input_shape[-1], 3, strides=(1, 1, 1), padding='same'))
        return layer_concat

    @staticmethod
    def conv_3d_transpose_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 \
               and input_shape[3] is not None and input_shape[3] >= 3

    @staticmethod
    def max_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.MaxPooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def average_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.AveragePooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def pooling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def max_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.MaxPooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def average_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.AveragePooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def pooling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def max_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.MaxPooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def average_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.AveragePooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def pooling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def batch_normalization_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.BatchNormalization(input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def batch_normalization_dense_input_legal(input_shape):
        return True

    @staticmethod
    def leaky_relu_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.LeakyReLU(input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def leaky_relu_dense_input_legal(input_shape):
        return True

    @staticmethod
    def prelu_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.PReLU(input_shape=input_shape[1:], alpha_initializer='RandomNormal'))
        return layer_concat

    @staticmethod
    def prelu_dense_input_legal(input_shape):
        return True

    @staticmethod
    def elu_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ELU(input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def elu_dense_input_legal(input_shape):
        return True

    @staticmethod
    def thresholded_relu_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ThresholdedReLU(input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def thresholded_relu_dense_input_legal(input_shape):
        return True

    @staticmethod
    def softmax_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Softmax(input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def softmax_dense_input_legal(input_shape):
        return True

    @staticmethod
    def relu_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ReLU(max_value=1.0, input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def relu_dense_input_legal(input_shape):
        return True

    @staticmethod
    def dropout_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Dropout(0.2, input_shape=input_shape[1:]))
        return layer_concat

    @staticmethod
    def dropout_dense_input_legal(input_shape):
        return True

    @staticmethod
    def timeDistributed_dense(input_shape):
        import keras
        layer_concat = []
        dense_layer = keras.layers.Dense(input_shape[-1])
        layer_concat.append(keras.layers.TimeDistributed(dense_layer, input_shape=input_shape))
        return layer_concat

    @staticmethod
    def timeDistributed_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[3] is not None

    @staticmethod
    def activity_normalization_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ActivityRegularization(0.5, 0.5))
        return layer_concat

    @staticmethod
    def activity_normalization_dense_input_legal(input_shape):
        return True

    @staticmethod
    def locally_Connected2D(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.LocallyConnected2D(input_shape[-1], 3, strides=(1, 1)))
        for layer in LayerMatching.reshape_block(input_shape, input_shape):
            layer_concat.append(layer)
        return layer_concat

    @staticmethod
    def locally_Connected2D_input_legal(input_shape):
        return len(input_shape) == 4 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] >= 3 \
               and input_shape[2] is not None and input_shape[2] >= 3 and LayerMatching.reshape_block_input_legal(input_shape)

    @staticmethod
    def locally_Connected1D(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.LocallyConnected1D(input_shape[-1], 3, strides=1))
        for layer in LayerMatching.reshape_block(input_shape, input_shape):
            layer_concat.append(layer)
        return layer_concat

    @staticmethod
    def locally_Connected1D_input_legal(input_shape):
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and LayerMatching.reshape_block_input_legal(input_shape)

    # JTJ在这里结束了扩展


if __name__ == '__main__':
    pass
