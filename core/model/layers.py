import mxnet as mx

# ==== layers ====

def Convolution(data, num_filter, kernel, stride=None, dilate=None, pad=None, num_group=1, no_bias=False,
                weight=None, bias=None, name=None, lr_mult=1, reuse=None, **kwargs):
    if reuse is not None:
        assert name is not None
    name = GetLayerName.get('conv') if name is None else name

    stride = (1,) * len(kernel) if stride is None else stride
    dilate = (1,) * len(kernel) if dilate is None else dilate

    # tensorflow 'SAME' padding
    if isinstance(pad, str):
        input_size = kwargs.get('input_size', None)
        if input_size is None:
            raise ValueError("`input_size` is needed for padding")
        del kwargs['input_size']
        if isinstance(input_size, int):
            in_size_h = in_size_w = input_size
        else:
            in_size_h, in_size_w = input_size
        ph0, ph1 = padding_helper(in_size_h, kernel[0], stride[0], pad)
        pw0, pw1 = padding_helper(in_size_w, kernel[1], stride[1], pad)
        data = mx.sym.pad(data, mode='constant', pad_width=(0,0,0,0,ph0,ph1,pw0,pw1))
        pad = (0,) * len(kernel)
    else:
        pad = (0,) * len(kernel) if pad is None else pad
    assert len(kwargs) == 0, sorted(kwargs)

    W = get_variable(name+'_weight', lr_mult, reuse) if weight is None else weight
    if no_bias:
        x = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, dilate=dilate, pad=pad,
                               num_group=num_group, no_bias=no_bias, name=name if reuse is None else None,
                               weight=W)
    else:
        B = get_variable(name+'_bias', lr_mult, reuse) if bias is None else bias
        x = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, dilate=dilate, pad=pad,
                               num_group=num_group, no_bias=no_bias, name=name if reuse is None else None,
                               weight=W, bias=B)
    return x

def Deconvolution(data, num_filter, kernel, stride=None, dilate=None, pad=None, adj=None, target_shape=None,
                  num_group=1, no_bias=False, weight=None, bias=None, name=None, lr_mult=1, reuse=None):
    if reuse is not None:
        assert name is not None
    name = GetLayerName.get('deconv') if name is None else name

    stride = (1,) * len(kernel) if stride is None else stride
    dilate = (1,) * len(kernel) if dilate is None else dilate
    pad = (0,) * len(kernel) if pad is None else pad
    adj = (0,) * len(kernel) if adj is None else adj
    target_shape = tuple([]) if target_shape is None else target_shape

    W = get_variable(name+'_weight', lr_mult, reuse) if weight is None else weight
    if no_bias:
        x = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel, stride=stride, dilate=dilate, pad=pad,
                                 adj=adj, target_shape=target_shape, num_group=num_group, no_bias=no_bias,
                                 name=name if reuse is None else None, weight=W)
    else:
        B = get_variable(name+'_bias', lr_mult, reuse) if bias is None else bias
        x = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel, stride=stride, dilate=dilate, pad=pad,
                                 adj=adj, target_shape=target_shape, num_group=num_group, no_bias=no_bias,
                                 name=name if reuse is None else None, weight=W, bias=B)
    return x

def FullyConnected(data, num_hidden, flatten=True, no_bias=False, weight=None, bias=None, name=None, lr_mult=1, reuse=None):
    if reuse is not None:
        assert name is not None
    name = GetLayerName.get('fc') if name is None else name

    W = get_variable(name+'_weight', lr_mult, reuse) if weight is None else weight
    if no_bias:
        x = mx.sym.FullyConnected(data, num_hidden=num_hidden, flatten=flatten, no_bias=no_bias, weight=W,
                                  name=name if reuse is None else None)
    else:
        B = get_variable(name+'_bias', lr_mult, reuse) if bias is None else bias
        x = mx.sym.FullyConnected(data, num_hidden=num_hidden, flatten=flatten, no_bias=no_bias, weight=W, bias=B,
                                  name=name if reuse is None else None)
    return x

def Relu(data, name=None):
    name = GetLayerName.get('relu') if name is None else name
    x = mx.sym.Activation(data, act_type='relu', name=name)
    return x

def LeakyRelu(data, slope=0.25, name=None):
    name = GetLayerName.get('leakyRelu') if name is None else name
    x = mx.sym.LeakyReLU(data, slope=slope, act_type='leaky', name=name)
    return x

def Tanh(data, name=None):
    name = GetLayerName.get('tanh') if name is None else name
    x = mx.sym.tanh(data, name=name)
    return x

def Swish(data, name=None):
    name = GetLayerName.get('swish') if name is None else name
    x = data * mx.sym.sigmoid(data)
    return x

def Pooling(data, kernel, stride=None, pad=None, pool_type='max', global_pool=False, name=None):
    name = GetLayerName.get('pool') if name is None else name

    stride = kernel if stride is None else stride
    pad = (0,) * len(kernel) if pad is None else pad

    x = mx.sym.Pooling(data, kernel=kernel, stride=stride, pad=pad, pool_type=pool_type,
                       global_pool=global_pool, name=name)
    return x

def Dropout(data, p, name=None):
    name = GetLayerName.get('drop') if name is None else name

    x = mx.sym.Dropout(data, p=p, name=name)
    return x

def BatchNorm(data, fix_gamma=False, momentum=0.9, eps=1e-5, use_global_stats=False, gamma=None, beta=None,
              moving_mean=None, moving_var=None, name=None, lr_mult=1, reuse=None):
    if reuse is not None:
        assert name is not None
    name = GetLayerName.get('bn') if name is None else name

    gamma = get_variable(name+'_gamma', lr_mult, reuse) if gamma is None else gamma
    beta = get_variable(name+'_beta', lr_mult, reuse) if beta is None else beta
    moving_mean = get_variable(name+'_moving_mean', 1, reuse) if moving_mean is None else moving_mean
    moving_var = get_variable(name+'_moving_var', 1, reuse) if moving_var is None else moving_var

    x = mx.sym.BatchNorm(data, fix_gamma=fix_gamma, momentum=momentum, eps=eps, use_global_stats=use_global_stats,
                         gamma=gamma, beta=beta, moving_mean=moving_mean, moving_var=moving_var,
                         name=name if reuse is None else None)
    return x

def InstanceNorm(data, eps=1e-5, gamma=None, beta=None, name=None, lr_mult=1, reuse=None):
    if reuse is not None:
        assert name is not None
    name = GetLayerName.get('in') if name is None else name

    gamma = get_variable(name+'_gamma', lr_mult, reuse) if gamma is None else gamma
    beta = get_variable(name+'_beta', lr_mult, reuse) if beta is None else beta

    x = mx.sym.InstanceNorm(data, eps=eps, gamma=gamma, beta=beta, name=name if reuse is None else None)
    return x

def Flatten(data, name=None):
    name = GetLayerName.get('flatten') if name is None else name
    x = mx.sym.flatten(data, name=name)
    return x

# ==== shortcuts ====
Conv = Convolution
Deconv = Deconvolution
FC = FullyConnected
Pool = Pooling
Drop = Dropout
BN = BatchNorm
IN = InstanceNorm

def ConvRelu(*args, **kwargs):
    x = Conv(*args, **kwargs)
    x = Relu(x, x.name+'_relu')
    return x

def BNRelu(*args, **kwargs):
    x = BN(*args, **kwargs)
    x = Relu(x, x.name+'_relu')
    return x

def FCRelu(*args, **kwargs):
    x = FC(*args, **kwargs)
    x = Relu(x, x.name+'_relu')
    return x

def ConvBNRelu(*args, **kwargs):
    x = Conv(*args, **kwargs)
    x = BN(x, name=x.name+'_bn', lr_mult=kwargs.get('lr_mult', 1), reuse=kwargs.get('reuse', None))
    x = Relu(x, x.name+'_relu')
    return x

# ==== __utils__ ====
def get_variable(name, lr_mult=1, reuse=None):
    if reuse is None:
        return mx.sym.Variable(name, lr_mult=lr_mult)
    return reuse.get_internals()[name]

class GetLayerName(object):
    _name_count = {}

    @classmethod
    def get(cls, name_prefix):
        cnt = cls._name_count.get(name_prefix, 0)
        cls._name_count[name_prefix] = cnt + 1
        return name_prefix + str(cnt)

def padding_helper(in_size, kernel_size, stride, pad_type='same'):
    pad_type = pad_type.lower()
    if pad_type == 'same':
        out_size = in_size // stride + int((in_size % stride) > 0)
        pad_size = max((out_size - 1) * stride + kernel_size - in_size, 0)
        return pad_size // 2, pad_size - pad_size // 2
    else:
        raise ValueError(pad_type)

