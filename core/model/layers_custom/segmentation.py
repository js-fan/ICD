from ..layers import *

# ==== Bilinear Sampling ====
class BilinearScale(mx.operator.CustomOp):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        h, w = x.shape[2:]
        new_h = int((h - 1) * self.scale) + 1
        new_w = int((w - 1) * self.scale) + 1

        x.attach_grad()
        with mx.autograd.record():
            new_x = mx.nd.contrib.BilinearResize2D(x, height=new_h, width=new_w)
        self.new_x = new_x 
        self.x = x

        self.assign(out_data[0], req[0], new_x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.new_x.backward(out_grad[0])
        self.assign(in_grad[0], req[0], self.x.grad)

@mx.operator.register("BilinearScale")
class BilinearScaleProp(mx.operator.CustomOpProp):
    def __init__(self, scale):
        super(BilinearScaleProp, self).__init__(need_top_grad=True)
        self.scale = float(scale)

    def infer_shape(self, in_shape):
        n, c, h, w = in_shape[0]
        new_h = int((h - 1) * self.scale) + 1
        new_w = int((w - 1) * self.scale) + 1
        return in_shape, [(n, c, new_h, new_w)], []

    def create_operator(self, ctx, shapes, dtypes):
        return BilinearScale(self.scale)

class BilinearScaleLike(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x, x_ref = in_data
        new_h, new_w = x_ref.shape[2:]

        x.attach_grad()
        with mx.autograd.record():
            new_x = mx.nd.contrib.BilinearResize2D(x, height=new_h, width=new_w)
        self.new_x = new_x
        self.x = x

        self.assign(out_data[0], req[0], new_x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.new_x.backward(out_grad[0])
        in_grad[1][:] = 0
        self.assign(in_grad[0], req[0], self.x.grad)

@mx.operator.register("BilinearScaleLike")
class BilinearScaleLikeProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(BilinearScaleLikeProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['d1', 'd2']

    def infer_shape(self, in_shape):
        out_shape = list(in_shape[1])
        out_shape[1] = in_shape[0][1]
        return in_shape, [out_shape,], []

    def create_operator(self, ctx, shapes, dtypes):
        return BilinearScaleLike()


# ==== Loss ====
class SegmentLoss(mx.operator.CustomOp):
    def __init__(self, has_grad_scale, onehot_label, grad_scale):
        self.has_grad_scale = has_grad_scale
        self.onehot_label = onehot_label
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        # logit, label, (grad_scale) = in_data
        prediction = mx.nd.softmax(in_data[0], axis=1)
        self.assign(out_data[0], req[0], prediction)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prediction = out_data[0]
        if not self.onehot_label:
            label = mx.nd.one_hot(in_data[1], depth=prediction.shape[1]).transpose((0, 3, 1, 2))
        else:
            label = in_data[1]

        if prediction.shape[2] != label.shape[2]:
            label = mx.nd.contrib.BilinearResize2D(label,
                    height=prediction.shape[2], width=prediction.shape[3])
            label = mx.nd.one_hot(mx.nd.argmax(label, axis=1),
                    depth=prediction.shape[1]).transpose((0, 3, 1, 2)) * (mx.nd.max(label, axis=1, keepdims=True) > 0.5)

        mask = label.sum(axis=1, keepdims=True)
        num_pixel = mx.nd.maximum(mask.sum() / mask.shape[0], 1)

        grad = (prediction - label) * mask / num_pixel
        if self.has_grad_scale:
            grad_scale = in_data[2].reshape(-1, 1, 1, 1)
            grad = grad * grad_scale

        grad = grad * self.grad_scale

        in_grad[1][:] = 0
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("SegmentLoss")
class SegmentLossProp(mx.operator.CustomOpProp):
    def __init__(self, has_grad_scale=0, onehot_label=0, grad_scale=1):
        super(SegmentLossProp, self).__init__(need_top_grad=False)
        self.has_grad_scale = int(has_grad_scale) > 0
        self.onehot_label = int(onehot_label) > 0
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        if self.has_grad_scale:
            return ['data', 'label', 'scale']
        else:
            return ['data', 'label']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0],], []

    def create_operator(self, ctx, shapes, dtypes):
        return SegmentLoss(self.has_grad_scale, self.onehot_label, self.grad_scale)

class MultiSigmoidLoss(mx.operator.CustomOp):
    def __init__(self, grad_scale):
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        logit, label = in_data
        prediction = mx.nd.sigmoid(logit, axis=1)
        self.assign(out_data[0], req[0], prediction)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prediction = out_data[0]
        label = in_data[1]

        grad = (prediction - label) * self.grad_scale
        #grad = prediction - label

        in_grad[1][:] = 0
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("MultiSigmoidLoss")
class MultiSigmoidLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1):
        super(MultiSigmoidLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiSigmoidLoss(self.grad_scale)

class MultiSoftmaxLoss(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        logit, label = in_data
        prediction = mx.nd.softmax(logit, axis=1)
        self.assign(out_data[0], req[0], prediction)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prediction = out_data[0]
        label = in_data[1]

        grad = prediction - label

        in_grad[1][:] = 0
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("MultiSoftmaxLoss")
class MultiSoftmaxLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MultiSoftmaxLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiSoftmaxLoss()

# ==== Others ====

