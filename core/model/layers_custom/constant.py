from ..layers import *
import numpy as np

class OpConstant(mx.operator.CustomOp):
    def __init__(self, val):
        self.val = val

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register('Constant')
class OpConstantProp(mx.operator.CustomOpProp):
    def __init__(self, val_str, shape_str, type_str='float32'):
        super(OpConstantProp, self).__init__(need_top_grad=False)
        val = [float(x) for x in val_str.split(',')]
        shape = [int(x) for x in shape_str.split(',')]
        self.val = mx.nd.array(val, dtype=type_str).reshape(shape)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [self.val.shape], []

    def infer_type(self, in_type):
        return in_type, [self.val.dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return OpConstant(self.val.as_in_context(ctx))

def CustomConstantEncoder(value, dtype='float32'):
    if not isinstance(value, np.ndarray):
        if not isinstance(value, (list, tuple)):
            value = [value]
        value = np.array(value, dtype=dtype)
    return ','.join([str(x) for x in value.ravel()]), ','.join([str(x) for x in value.shape])


def Constant(value, dtype='float32'):
    assert isinstance(dtype, str), dtype
    val, shape = CustomConstantEncoder(value, dtype)
    return mx.sym.Custom(val_str=val, shape_str=shape, type_str=dtype, op_type='Constant')

