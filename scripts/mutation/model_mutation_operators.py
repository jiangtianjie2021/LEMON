from scripts.tools import utils
import math
from typing import *
from scripts.mutation.mutation_utils import *
from scripts.mutation.layer_matching import LayerMatching
import random
import os
import warnings
from scripts.logger.lemon_logger import Logger
import keras
import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

mylogger = Logger()


def _assert_indices(mutated_layer_indices: List[int], depth_layer: int):
    assert max(mutated_layer_indices) < depth_layer, "Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0, "Min index should be greater than or equal to zero"


def _shuffle_conv3d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            filter_width, filter_height, filter_beight, num_of_input_channels, num_of_output_channels = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,
                                       (filter_width * filter_height * filter_beight * num_of_input_channels,
                                        num_of_output_channels))
                selected_list = copy_list[:, output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,
                                 (filter_width, filter_height, filter_beight, num_of_input_channels,
                                  num_of_output_channels))
        new_weights.append(val)
    return new_weights


def _shuffle_conv2d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,
                                       (filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:, output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,
                                 (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        new_weights.append(val)
    return new_weights


def _shuffle_depthwise_conv2d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
            assert num_of_output_channels == 1
            mutate_iutput_channels = utils.ModelUtils.generate_permutation(num_of_input_channels, mutate_ratio)
            for iutput_channel in mutate_iutput_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,
                                       (filter_width * filter_height * num_of_output_channels, num_of_input_channels))
                selected_list = copy_list[:, iutput_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, iutput_channel] = shuffle_selected_list
                val = np.reshape(copy_list,
                                 (filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        new_weights.append(val)
    return new_weights


def _shuffle_conv1d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            filter, num_of_input_channels, num_of_output_channels = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,
                                       (filter * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:, output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,
                                 (filter, num_of_input_channels, num_of_output_channels))
        new_weights.append(val)
    return new_weights


def _shuffle_batch_normal(weights, mutate_ratio):
    new_weights = []
    val = np.array(weights)
    # for val in weights:
    # val is bias if len(val.shape) == 1
    if len(val.shape) > 1:
        val_shape = val.shape
        input_dim, output_dim = val_shape
        mutate_output_dims = utils.ModelUtils.generate_permutation(output_dim, mutate_ratio)
        copy_list = val.copy()
        for output_dim in mutate_output_dims:
            selected_list = copy_list[:, output_dim]
            shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
            copy_list[:, output_dim] = shuffle_selected_list
        val = copy_list

    for v in val:
        new_weights.append(v)

    return new_weights


def _shuffle_dense(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            input_dim, output_dim = val_shape
            mutate_output_dims = utils.ModelUtils.generate_permutation(output_dim, mutate_ratio)
            copy_list = val.copy()
            for output_dim in mutate_output_dims:
                selected_list = copy_list[:, output_dim]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_dim] = shuffle_selected_list
            val = copy_list
        new_weights.append(val)
    return new_weights


def _LA_model_scan(model, new_layers, mutated_layer_indices=None):
    layer_utils = LayerUtils()
    layers = model.layers
    # new layers can never be added after the last layer
    positions_to_add = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(layers))

    insertion_points = {}
    available_new_layers = [layer for layer in
                            layer_utils.available_model_level_layers.keys()] if new_layers is None else new_layers
    # available_new_layers = ["dropout"] if new_layers is None else new_layers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in positions_to_add:
            for available_new_layer in available_new_layers:
                if layer_utils.is_input_legal[available_new_layer](layer.output.shape):
                    if i not in insertion_points.keys():
                        insertion_points[i] = [available_new_layer]
                    else:
                        insertion_points[i].append(available_new_layer)
    return insertion_points


def _MergLA_model_scan(model, new_layers, mutated_layer_indices=None):
    layer_utils = LayerUtils()
    layers = model.layers
    # new layers can never be added after the last layer
    positions_to_add = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(layers))

    insertion_points = {}
    result = {}
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in positions_to_add:
            output_shape = str(layer.output_shape)
            if output_shape not in insertion_points.keys():
                insertion_points[output_shape] = [layer.name]
            else:
                insertion_points[output_shape].append(layer.name)
    for key in insertion_points.keys():
        if len(insertion_points[key]) > 1:
            result[key] = insertion_points[key]
    return result


def _MLA_model_scan(model, new_layers, mutated_layer_indices=None):
    layer_matching = LayerMatching()
    layers = model.layers
    # new layers can never be added after the last layer
    positions_to_add = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(layers))

    insertion_points = {}
    available_new_layers = [layer for layer in
                            layer_matching.layer_concats.keys()] if new_layers is None else new_layers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in positions_to_add:
            for available_new_layer in available_new_layers:
                # print('{} test shape: {} as list: {}'.format(available_new_layer, layer.output.shape,
                #                                              layer.output.shape.as_list()))
                if layer_matching.input_legal[available_new_layer](layer.output.shape):
                    # print('shape {} can be inserted'. format(layer.output.shape))
                    if i not in insertion_points.keys():
                        insertion_points[i] = [available_new_layer]
                    else:
                        insertion_points[i].append(available_new_layer)
    return insertion_points


def _LC_and_LR_scan(model, mutated_layer_indices):
    layers = model.layers

    # the last layer should not be copied or removed
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(mutated_layer_indices, len(layers))

    available_layer_indices = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in mutated_layer_indices:
            # InputLayer should not be copied or removed
            from keras.engine.input_layer import InputLayer
            if isinstance(layer, InputLayer):
                continue
            # layers with multiple input tensors can't be copied or removed
            if isinstance(layer.input, list) and len(layer.input) > 1:
                continue
            layer_input_shape = layer.input.shape.as_list()
            layer_output_shape = layer.output.shape.as_list()
            # if layer_input_shape == layer_output_shape:
            if (LayerMatching.reshape_block_input_legal(layer.input_shape) and LayerMatching.reshape_block_input_legal(layer.output_shape)) or layer_input_shape == layer_output_shape:
                available_layer_indices.append(i)
    np.random.shuffle(available_layer_indices)
    return available_layer_indices


def _LS_scan(model):
    layers = model.layers
    shape_dict = {}
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if isinstance(layer.input, list) and len(layer.input) > 1:
            continue
        layer_input_shape = [str(i) for i in layer.input.shape.as_list()[1:]]
        layer_output_shape = [str(i) for i in layer.output.shape.as_list()[1:]]
        input_shape = "-".join(layer_input_shape)
        output_shape = "-".join(layer_output_shape)
        k = "+".join([input_shape, output_shape])
        if k not in shape_dict.keys():
            shape_dict[k] = [i]
        else:
            shape_dict[k].append(i)
    return shape_dict


def GF_mut(model, mutation_ratio, distribution='normal', STD=0.1, lower_bound=None, upper_bound=None):
    # distribution感觉没用上
    # valid_distributions = ['normal', 'uniform']
    # assert distribution in valid_distributions, 'Distribution %s is not support.' % distribution
    # if distribution == 'uniform' and (lower_bound is None or upper_bound is None):
    #     mylogger.error('Lower bound and Upper bound is required for uniform distribution.')
    #     raise ValueError('Lower bound and Upper bound is required for uniform distribution.')

    mylogger.info('copying model...')
    GF_model = utils.ModelUtils.model_copy(model, 'GF')
    mylogger.info('model copied')

    layers = GF_model.layers
    mutated_layer_indices = np.arange(len(layers))
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        layer_allow = mutated_layer_indices[0]
        for i in mutated_layer_indices:
            layer = layers[i]
            # skip if layer is not in white list
            if layer_utils.is_layer_in_weight_change_white_list_fix(layer):
                layer_allow = i
                break
        layer = layers[layer_allow]
        mylogger.info('executing mutation of {}'.format(layer.name))
        weights = layer.get_weights()
        new_weights = []
        for weight in weights:
            weight_shape = weight.shape
            weight_flat = weight.flatten()
            permu_num = math.floor(len(weight_flat) * mutation_ratio)
            permutation = np.random.permutation(len(weight_flat))[:permu_num]
            STD = math.sqrt(weight_flat.var()) * STD
            weight_flat[permutation] += np.random.normal(scale=STD, size=len(permutation))
            weight = weight_flat.reshape(weight_shape)
            new_weights.append(weight)
        layer.set_weights(new_weights)

    return GF_model


# 扩展WS算子只用于两个层
def WS_mut(model, mutation_ratio, mutated_layer_indices=None):
    mylogger.info('copying model...')
    WS_model = utils.ModelUtils.model_copy(model, 'WS')
    mylogger.info('model copied')

    layers = WS_model.layers
    mutated_layer_indices = np.arange(len(layers)) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        layer_allow = mutated_layer_indices[0]
        for i in mutated_layer_indices:
            layer = layers[i]
            # skip if layer is not in white list
            if layer_utils.is_layer_in_weight_change_white_list_fix(layer):
                layer_allow = i
                break
        layer = layers[layer_allow]
        mylogger.info('executing mutation of {}'.format(layer.name))
        weights = layer.get_weights()
        layer_name = type(layer).__name__
        # keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D, keras.layers.DepthwiseConv2D, keras.layers.Conv2DTranspose, keras.layers.Conv3DTranspose,
        if layer_name == "DepthwiseConv2D" and len(weights) != 0:
            layer.set_weights(_shuffle_depthwise_conv2d(weights, mutation_ratio))
        elif "Conv1D" in layer_name and len(weights) != 0:
            layer.set_weights(_shuffle_conv1d(weights, mutation_ratio))
        elif "Conv2D" in layer_name and len(weights) != 0:
            layer.set_weights(_shuffle_conv2d(weights, mutation_ratio))
        elif "Conv3D" in layer_name and len(weights) != 0:
            layer.set_weights(_shuffle_conv3d(weights, mutation_ratio))
        elif (layer_name == "Dense") and len(weights) != 0:
            layer.set_weights(_shuffle_dense(weights, mutation_ratio))
        elif (layer_name == "BatchNormalization") and len(weights) != 0:
            layer.set_weights(_shuffle_batch_normal(weights, mutation_ratio))
        else:
            pass
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")

    return WS_model


def NEB_mut(model, mutation_ratio, mutated_layer_indices=None):
    mylogger.info('copying model...')
    NEB_model = utils.ModelUtils.model_copy(model, 'NEB')
    mylogger.info('model copied')

    layers = NEB_model.layers
    mutated_layer_indices = np.arange(len(layers)) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        layer_allow = mutated_layer_indices[0]
        for i in mutated_layer_indices:
            layer = layers[i]
            # skip if layer is not in white list
            if layer_utils.is_layer_in_weight_change_white_list_fix(layer):
                layer_allow = i
                break
        layer = layers[layer_allow]
        mylogger.info('executing mutation of {}'.format(layer.name))
        weights = layer.get_weights()
        layer_name = type(layer).__name__
        assert isinstance(weights, list)
        if layer_name == "DepthwiseConv2D":
            weights_w = weights[0]
            filter_width, filter_height, num_of_input_channels, num_of_output_channels = weights_w.shape
            permutation = utils.ModelUtils.generate_permutation(num_of_input_channels, mutation_ratio)
            weights_w = np.reshape(weights_w,
                                   (filter_width * filter_height * num_of_output_channels,
                                    num_of_input_channels))
            # selected_list = copy_list[:, permutation]
            weights_w = weights_w.transpose()
            # weights_w[permutation] = np.zeros((len(permutation), weights_w[:, 0].shape[0]))
            weights_w[permutation] = np.zeros(weights_w[0].shape)
            weights_w = weights_w.transpose()
            weights_w = np.reshape(weights_w,
                                   (filter_width, filter_height, num_of_input_channels,
                                    num_of_output_channels))
            weights = [weights_w]
            layer.set_weights(weights)
        elif len(weights) == 1:
            weights_w = weights[0]
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] = np.zeros(weights_w[0].shape)
            weights_w = weights_w.transpose()
            weights = [weights_w]
            layer.set_weights(weights)
        elif len(weights) == 2:
            weights_w, weights_b = weights
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] = np.zeros(weights_w[0].shape)
            weights_w = weights_w.transpose()
            weights_b[permutation] = 0
            weights = weights_w, weights_b
            layer.set_weights(weights)
        elif layer_name == "BatchNormalization":
            weights_w = np.array(weights)
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] = np.zeros(weights_w[0].shape)
            weights_w = weights_w.transpose()
            weights = []
            for w in weights_w:
                weights.append(w)
            layer.set_weights(weights)
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return NEB_model


def NAI_mut(model, mutation_ratio, mutated_layer_indices=None):
    mylogger.info('copying model...')
    NAI_model = utils.ModelUtils.model_copy(model, 'NAI')
    mylogger.info('model copied')

    layers = NAI_model.layers
    mutated_layer_indices = np.arange(len(layers)) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        layer_allow = mutated_layer_indices[0]
        for i in mutated_layer_indices:
            layer = layers[i]
            # skip if layer is not in white list
            if layer_utils.is_layer_in_weight_change_white_list_fix(layer):
                layer_allow = i
                break
        layer = layers[layer_allow]
        mylogger.info('executing mutation of {}'.format(layer.name))
        weights = layer.get_weights()
        layer_name = type(layer).__name__
        assert isinstance(weights, list)
        if layer_name == "DepthwiseConv2D":
            weights_w = weights[0]
            filter_width, filter_height, num_of_input_channels, num_of_output_channels = weights_w.shape
            permutation = utils.ModelUtils.generate_permutation(num_of_input_channels, mutation_ratio)
            weights_w = np.reshape(weights_w,
                                   (filter_width * filter_height * num_of_output_channels,
                                    num_of_input_channels))
            weights_w = weights_w.transpose()
            weights_w[permutation] *= -1
            weights_w = weights_w.transpose()
            weights_w = np.reshape(weights_w,
                                   (filter_width, filter_height, num_of_input_channels,
                                    num_of_output_channels))
            weights = [weights_w]
            layer.set_weights(weights)
        elif len(weights) == 1:
            weights_w = weights[0]
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] *= -1
            weights_w = weights_w.transpose()
            weights = [weights_w]
            layer.set_weights(weights)
        elif len(weights) == 2:
            weights_w, weights_b = weights
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] *= -1
            weights_w = weights_w.transpose()
            weights_b[permutation] *= -1
            weights = weights_w, weights_b
            layer.set_weights(weights)
        elif layer_name == "BatchNormalization":
            weights_w = np.array(weights)
            weights_w = weights_w.transpose()
            permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
            weights_w[permutation] *= -1
            weights_w = weights_w.transpose()
            weights = []
            for w in weights_w:
                weights.append(w)
            layer.set_weights(weights)
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return NAI_model


def NS_mut(model, mutated_layer_indices=None):
    mylogger.info('copying model...')
    NS_model = utils.ModelUtils.model_copy(model, 'NS')
    mylogger.info('model copied')

    layers = NS_model.layers
    mutated_layer_indices = np.arange(len(layers)) if mutated_layer_indices is None else mutated_layer_indices

    _assert_indices(mutated_layer_indices, len(layers))
    layer_utils = LayerUtils()
    np.random.shuffle(mutated_layer_indices)
    layer_allow = mutated_layer_indices[0]
    for i in mutated_layer_indices:
        layer = layers[i]
        # skip if layer is not in white list
        if layer_utils.is_layer_in_weight_change_white_list_fix(layer):
            layer_allow = i
            break
    layer = layers[layer_allow]
    mylogger.info('executing mutation of {}'.format(layer.name))
    weights = layer.get_weights()
    layer_name = type(layer).__name__
    assert isinstance(weights, list)
    if layer_name == "DepthwiseConv2D":
        weights_w = weights[0]
        filter_width, filter_height, num_of_input_channels, num_of_output_channels = weights_w.shape
        weights_w = np.reshape(weights_w,
                               (filter_width * filter_height * num_of_output_channels,
                                num_of_input_channels))
        weights_w = weights_w.transpose()
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]

            weights_w[permutation[0]], weights_w[permutation[1]] = \
                weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
            weights_w = weights_w.transpose()
            weights_w = np.reshape(weights_w,
                                   (filter_width, filter_height, num_of_input_channels,
                                    num_of_output_channels))
            weights = [weights_w]

            layer.set_weights(weights)
        else:
            mylogger.warning("NS not used! One neuron can't be shuffle!")
    elif len(weights) == 1:
        weights_w = weights[0]
        weights_w = weights_w.transpose()
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]

            weights_w[permutation[0]], weights_w[permutation[1]] = \
                weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
            weights_w = weights_w.transpose()
            weights = [weights_w]

            layer.set_weights(weights)
        else:
            mylogger.warning("NS not used! One neuron can't be shuffle!")
    elif len(weights) == 2:
        weights_w, weights_b = weights
        weights_w = weights_w.transpose()
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]

            weights_w[permutation[0]], weights_w[permutation[1]] = \
                weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
            weights_w = weights_w.transpose()

            weights_b[permutation[0]], weights_b[permutation[1]] = \
                weights_b[permutation[1]].copy(), weights_b[permutation[0]].copy()

            weights = weights_w, weights_b

            layer.set_weights(weights)
        else:
            mylogger.warning("NS not used! One neuron can't be shuffle!")
    elif layer_name == "BatchNormalization":
        weights_w = np.array(weights)
        weights_w = weights_w.transpose()
        if weights_w.shape[0] >= 2:
            permutation = np.random.permutation(weights_w.shape[0])[:2]

            weights_w[permutation[0]], weights_w[permutation[1]] = \
                weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
            weights_w = weights_w.transpose()
            weights = []
            for w in weights_w:
                weights.append(w)
            layer.set_weights(weights)
        else:
            mylogger.warning("NS not used! One neuron can't be shuffle!")

    return NS_model


def ARem_mut(model, mutated_layer_indices=None):
    ARem_model = utils.ModelUtils.model_copy(model, 'ARem')
    layers = ARem_model.layers
    # the activation of last layer should not be removed
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))

    for i in mutated_layer_indices:
        layer = layers[i]
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = ActivationUtils.no_activation
            break
    return ARem_model


def ARep_mut(model, new_activations=None, mutated_layer_indices=None):
    activation_utils = ActivationUtils()
    ARep_model = utils.ModelUtils.model_copy(model, 'ARep')
    layers = ARep_model.layers
    # the activation of last layer should not be replaced
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))
    for i in mutated_layer_indices:
        layer = layers[i]
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = activation_utils.pick_activation_randomly(new_activations)
            break
    return ARep_model


def LA_mut(model, new_layers=None, mutated_layer_indices=None):
    layer_utils = LayerUtils()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_utils.available_model_level_layers.keys():
                mylogger.error('Layer {} is not supported.'.format(layer))
                raise Exception('Layer {} is not supported.'.format(layer))
    LA_model = utils.ModelUtils.model_copy(model, 'LA')

    insertion_points = _LA_model_scan(LA_model, new_layers, mutated_layer_indices)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                      .format(insertion_points[key], key, type(model.layers[key])))
    layers_index_avaliable = list(insertion_points.keys())

    if model.__class__.__name__ != 'Sequential':
        layers_index_avaliable_len = len(layers_index_avaliable) - 1
    else:
        layers_index_avaliable_len = len(layers_index_avaliable)
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, layers_index_avaliable_len)]

    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('insert {} after {}'.format(layer_name_to_insert, LA_model.layers[layer_index_to_insert].name))
    # insert new layer
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LA_model.layers):
            new_layer = LayerUtils.clone(layer)
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output_shape
                new_model.add(layer_utils.available_model_level_layers[layer_name_to_insert](output_shape))
    else:

        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output_shape
            new_layer = layer_utils.available_model_level_layers[layer_name_to_insert](output_shape)
            x = new_layer(x)
            return x

        new_model = utils.ModelUtils.functional_model_operation(LA_model, operation={
            LA_model.layers[layer_index_to_insert + 1].name: layer_addition})

    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    import time
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        if layer_name.endswith('_copy_LA'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def EmbLA_mut(model, new_layers=None, mutated_layer_indices=None):
    EmbLA_model = utils.ModelUtils.model_copy(model, 'EmbLA')
    if LayerMatching.embedding_input_legal(model.input.shape):
        import keras
        new_model = keras.models.Sequential()
        from keras.engine.input_layer import InputLayer
        if isinstance(EmbLA_model.layers[0], InputLayer):
            new_layer = LayerUtils.clone(EmbLA_model.layers[0])
            new_model.add(new_layer)
        new_model.add(LayerMatching.embedding_dense(EmbLA_model.input_shape))
        for i, layer in enumerate(EmbLA_model.layers):
            if isinstance(layer, InputLayer):
                continue
            new_layer = LayerUtils.clone(layer)
            new_model.add(new_layer)
        new_model.build(EmbLA_model.input_shape)
    else:
        return None

    if isinstance(EmbLA_model.layers[0], InputLayer):
        assert len(new_model.layers) == len(model.layers)
    else:
        assert len(new_model.layers) == len(model.layers) + 1

    tuples = []

    old_model_layers = {}
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            continue
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        if layer_name.endswith('_copy_EmbLA'):
            key = layer_name[:-11]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def MLA_mut(model, new_layers=None, mutated_layer_indices=None):
    # mutiple layers addition
    layer_matching = LayerMatching()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    MLA_model = utils.ModelUtils.model_copy(model, 'MLA')
    insertion_points = _MLA_model_scan(model, new_layers, mutated_layer_indices)
    mylogger.info(insertion_points)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                      .format(insertion_points[key], key, type(model.layers[key])))

    # use logic: randomly select a new layer available to insert into the layer which can be inserted
    layers_index_avaliable = list(insertion_points.keys())
    # layer_index_to_insert = np.max([i for i in insertion_points.keys()])
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info(
        'choose to insert {} after {}'.format(layer_name_to_insert, MLA_model.layers[layer_index_to_insert].name))
    # insert new layers
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(MLA_model.layers):
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += "_copy"
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output.shape.as_list()
                layers_to_insert = layer_matching.layer_concats[layer_name_to_insert](output_shape)
                for layer_to_insert in layers_to_insert:
                    layer_to_insert.name += "_insert"
                    mylogger.info(layer_to_insert)
                    new_model.add(layer_to_insert)
        new_model.build(MLA_model.input_shape)
    else:
        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output.shape.as_list()
            new_layers = layer_matching.layer_concats[layer_name_to_insert](output_shape)
            for l in new_layers:
                l.name += "_insert"
                mylogger.info('insert layer {}'.format(str(l)))
                x = l(x)
            return x

        new_model = utils.ModelUtils.functional_model_operation(MLA_model, operation={
            MLA_model.layers[layer_index_to_insert].name: layer_addition})

    tuples = []
    import time
    start_time = time.time()

    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_MLA'):
            key = layer_name[:-9]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    end_time = time.time()
    print('set weight cost {}'.format(end_time - start_time))

    return new_model


def MergLA_mut(model, new_layers=None, mutated_layer_indices=None):

    layer_utils = MergeUtils()
    merge_layers = list(layer_utils.available_model_merge_layers.keys())
    merge_layer = merge_layers[np.random.randint(0, len(merge_layers))]

    MergLA_model = utils.ModelUtils.model_copy(model, 'MergLA')
    insertion_points = _MergLA_model_scan(MergLA_model, new_layers, mutated_layer_indices)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be merged with shape {}'
                      .format(insertion_points[key], key))
    layers_index_avaliable = list(insertion_points.keys())
    # 选出来打算在哪个输出层形状上merge
    layer_shape_to_merge = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    # 拿到输出形状对应的层
    available_old_layers = insertion_points[layer_shape_to_merge]
    # 从那个输出形状上，选择两个层，保存二者的index和name,shuffle,然后选择前两个index，再把名字取出来
    np.random.shuffle(available_old_layers)
    layers_name_to_insert = available_old_layers[:2]
    # layer_name_to_insert = available_old_layers[np.random.randint(0, len(available_old_layers))]
    # 也可以声明一下嘛
    mylogger.info('{} merge with {}'.format(layers_name_to_insert, merge_layer))

    from scripts.mutation.mutation_utils import LayerUtils
    input_layers = {}
    output_tensors = {}
    model_output = None
    import keras
    MergLA_model_i = keras.Model(MergLA_model.input, MergLA_model.output)
    for layer in MergLA_model_i.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in input_layers.keys():
                input_layers[layer_name] = [layer.name]
            else:
                input_layers[layer_name].append(layer.name)
    # 第一个问题，模型上面应该再加一个input
    output_tensors[MergLA_model_i.layers[0].name] = MergLA_model_i.input
    o1, o2 = None, None
    for layer in MergLA_model_i.layers[1:]:
        layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
        if len(layer_input_tensors) == 1:
            layer_input_tensors = layer_input_tensors[0]
        cloned_layer = LayerUtils.clone(layer)
        # if suffix is not None:
        #     cloned_layer.name += suffix
        x = cloned_layer(layer_input_tensors)

        if layer.name in layers_name_to_insert:
            if o1 == None:
                o1 = cloned_layer.output
            else:
                import keras
                o2 = cloned_layer.output
                x = layer_utils.available_model_merge_layers[merge_layer]([o1, o2])

        output_tensors[layer.name] = x
        model_output = x

    new_model = keras.Model(inputs=MergLA_model_i.inputs, outputs=model_output)

    assert len(new_model.layers) == len(MergLA_model_i.layers) + 1
    tuples = []
    import time
    old_model_layers = {}
    for layer in MergLA_model_i.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        # if layer_name.endswith('_copy_MergLA'):
        #     key = layer_name[:-12]
        # elif layer_name.endswith('_copy_MergLA_input'):
        #     key = layer_name[:-18]
        # else:
        key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def LC_mut(model, mutated_layer_indices=None):
    LC_model = utils.ModelUtils.model_copy(model, 'LC')
    available_layer_indices = _LC_and_LR_scan(LC_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to copy (input and output shape should be same)')
        return None

    # use logic: copy the last available layer
    copy_layer_index = available_layer_indices[-1]
    copy_layer_name = LC_model.layers[copy_layer_index].name + '_repeat'

    mylogger.info('choose to copy layer {}'.format(LC_model.layers[copy_layer_index].name))

    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LC_model.layers):
            new_model.add(LayerUtils.clone(layer))
            if i == copy_layer_index:
                if layer.input_shape != layer.output_shape:
                    for lay in LayerMatching.reshape_block(layer.output_shape, layer.input_shape):
                        lay.name += '_reshape_block'
                        new_model.add(lay)
                copy_layer = LayerUtils.clone(layer)
                copy_layer.name += '_repeat'
                new_model.add(copy_layer)
        new_model.build(LC_model.input_shape)
    else:
        def layer_repeat(x, layer):
            import keras
            x = layer(x)
            # flatten = keras.layers.Flatten()
            if layer.input_shape != layer.output_shape:
                if (len(layer.output_shape) > 2):
                    x = keras.layers.Flatten()(x)
                units = 1
                for i in range(len(layer.input_shape)):
                    if i == 0:
                        continue
                    units *= layer.input_shape[i]
                x = keras.layers.Dense(units)(x)
                x = keras.layers.Reshape(layer.input_shape[1:])(x)
            copy_layer = LayerUtils.clone(layer)
            copy_layer.name += '_repeat'
            x = copy_layer(x)
            return x

        new_model = utils.ModelUtils.functional_model_operation(LC_model, operation={
            LC_model.layers[copy_layer_index].name: layer_repeat})

    # update weights
    # assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LC'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        if layer_name + '_copy_LC_repeat' == copy_layer_name:
            for sw, w in zip(new_model_layers[copy_layer_name].weights, layer_weights):
                shape_sw = np.shape(sw)
                shape_w = np.shape(w)
                assert len(shape_sw) == len(shape_w)
                assert shape_sw[0] == shape_w[0]
                tuples.append((sw, w))

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def LR_mut(model, mutated_layer_indices=None):
    LR_model = utils.ModelUtils.model_copy(model, 'LR')
    available_layer_indices = _LC_and_LR_scan(LR_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to remove (input and output shape should be same)')
        return None

    # use logic: remove the last available layer
    remove_layer_index = available_layer_indices[-1]
    mylogger.info('choose to remove layer {}'.format(LR_model.layers[remove_layer_index].name))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LR_model.layers):
            if i != remove_layer_index:
                new_layer = LayerUtils.clone(layer)
                # new_layer.name += '_copy'
                new_model.add(new_layer)
            else:
                if layer.input_shape != layer.output_shape:
                    for lay in LayerMatching.reshape_block(layer.input_shape, layer.output_shape):
                        lay.name += '_reshape_block'
                        new_model.add(lay)
        new_model.build(LR_model.input_shape)
    else:
        def layer_remove(x, layer):
            import keras
            y = layer(x)
            # flatten = keras.layers.Flatten()
            if layer.input_shape != layer.output_shape:
                if (len(layer.input_shape) > 2):
                    x = keras.layers.Flatten()(x)
                units = 1
                for i in range(len(layer.output_shape)):
                    if i == 0:
                        continue
                    units *= layer.output_shape[i]
                x = keras.layers.Dense(units)(x)
                x = keras.layers.Reshape(layer.output_shape[1:])(x)
            # copy_layer = LayerUtils.clone(layer)
            # copy_layer.name += '_repeat'
            # x = copy_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LR_model, operation={
            LR_model.layers[remove_layer_index].name: layer_remove})

    # update weights
    # assert len(new_model.layers) == len(model.layers) - 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LR'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in new_model_layers.keys():
        if layer_name.endswith('_reshape_block'):
            continue
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def LS_mut(model):
    LS_model = utils.ModelUtils.model_copy(model, "LS")
    shape_dict = _LS_scan(LS_model)
    layers = LS_model.layers

    swap_list = []
    for v in shape_dict.values():
        if len(v) > 1:
            swap_list.append(v)
    if len(swap_list) == 0:
        mylogger.warning("No layers to swap!")
        return None
    swap_list = swap_list[random.randint(0, len(swap_list) - 1)]
    choose_index = random.sample(swap_list, 2)
    mylogger.info('choose to swap {} ({} - {}) and {} ({} - {})'.format(layers[choose_index[0]].name,
                                                                        layers[choose_index[0]].input.shape,
                                                                        layers[choose_index[0]].output.shape,
                                                                        layers[choose_index[1]].name,
                                                                        layers[choose_index[1]].input.shape,
                                                                        layers[choose_index[1]].output.shape))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.Sequential()
        for i, layer in enumerate(layers):
            if i == choose_index[0]:
                new_model.add(LayerUtils.clone(layers[choose_index[1]]))
            elif i == choose_index[1]:
                new_model.add(LayerUtils.clone(layers[choose_index[0]]))
            else:
                new_model.add(LayerUtils.clone(layer))
    else:
        layer_1 = layers[choose_index[0]]
        layer_2 = layers[choose_index[1]]
        new_model = utils.ModelUtils.functional_model_operation(LS_model, {
            layer_1.name: lambda x, layer: LayerUtils.clone(layer_2)(x),
            layer_2.name: lambda x, layer: LayerUtils.clone(layer_1)(x)})

    # update weights
    assert len(new_model.layers) == len(model.layers)
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LS'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


if __name__ == '__main__':
    pass
