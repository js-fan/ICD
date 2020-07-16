from ..layers import *
import numpy as np

@mx.operator.register('ICD_BottomUp')
class ICD_BottomUpProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1):
        super(ICD_BottomUpProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)

    def list_arguments(self):
        return ['data', 'label', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        n, c, h, w = in_shape[0]
        _, C = in_shape[1]
        in_shape[2] = [C, c]
        return in_shape, [[n, C, h, w]], []

    def infer_type(self, in_type):
        _type = in_type[0]
        return [_type]*3, [_type], []

    def create_operator(self, ctx, shapes, dtypes):
        return ICD_BottomUp(self.grad_scale)

class ICD_BottomUp(mx.operator.CustomOp):
    def __init__(self, grad_scale):
        self.grad_scale = grad_scale

    def forward(self, is_train, req, in_data, out_data, aux):
        data, label, weight = in_data

        weight_normed = mx.nd.L2Normalization(weight, mode='instance')
        dot = mx.nd.Convolution(data, weight=weight_normed.reshape(0, 0, 1, 1),
                num_filter=weight.shape[0], stride=(1, 1), kernel=(1, 1), no_bias=True)
        self.assign(out_data[0], req[0], dot)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data, label, weight = in_data
        dot = out_data[0]
        n, c, h, w = data.shape
        label = label * (label.sum(axis=1, keepdims=True) < 1.001)

        data_flat = data.transpose((1, 0, 2, 3)).reshape(0, -1)
        dot_flat = (dot * label.reshape(0, 0, 1, 1)).transpose((1, 0, 2, 3)).reshape(0, -1)
        grad = - mx.nd.dot(dot_flat, data_flat, transpose_b=True) * (self.grad_scale / (h * w))

        w_sum_square = (weight**2).sum(axis=1, keepdims=True)
        w_sum_sqrt = mx.nd.sqrt(w_sum_square)
        w_sum = weight.sum(axis=1, keepdims=True)
        wg_sum = (weight * grad).sum(axis=1, keepdims=True)
        grad = -weight * (wg_sum / (w_sum_square * w_sum_sqrt + 1e-5)) + grad / (w_sum_sqrt + 1e-5)

        in_grad[0][:] = 0
        in_grad[1][:] = 0
        self.assign(in_grad[2], req[2], grad)

@mx.operator.register('AccumResign')
class AccumResignProp(mx.operator.CustomOpProp):
    def __init__(self, momentum=0.9, use_global_stats=0):
        super(AccumResignProp, self).__init__(need_top_grad=False)
        self.momentum = float(momentum)
        self.use_global_stats = int(use_global_stats) > 0
        self.kvstore = None

    def list_arguments(self):
        return ['score', 'cam', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        return ['moving_mean']

    def infer_shape(self, in_shape):
        n, c, h, w = in_shape[0]
        self._moving_mean_shape = (c, 2)
        return in_shape, [in_shape[0]], [[c, 2]]

    @staticmethod
    def create_updater(momentum):
        def wrapper(key, inputs, stored):
            (stored*momentum + inputs.as_in_context(stored.context)*(1 - momentum)).copyto(stored)
        return wrapper

    def create_operator(self, ctx, shapes, dtypes):
        if self.kvstore is None:
            self.kvstore = mx.kv.create('device')
            updater = self.create_updater(self.momentum)
            self.kvstore._set_updater(updater)
            self.kvstore.init(0, mx.nd.zeros(self._moving_mean_shape, ctx=ctx))
        return AccumResign(self.kvstore, self.use_global_stats)

class AccumResign(mx.operator.CustomOp):
    def __init__(self, kvstore, use_global_stats):
        self.kvstore = kvstore
        self.use_global_stats = use_global_stats

    def forward(self, is_train, req, in_data, out_data, aux):
        score, cam, label = in_data
        accum_score = aux[0]

        if is_train and (not self.use_global_stats):
            pmask = score > 0
            nmask = score < 0

            cam = mx.nd.maximum(cam, 0)
            cam_pos = (cam * pmask).sum(axis=(2, 3)) / mx.nd.maximum(pmask.sum(axis=(2, 3)), 1e-5)
            cam_neg = (cam * nmask).sum(axis=(2, 3)) / mx.nd.maximum(nmask.sum(axis=(2, 3)), 1e-5)

            label_cnt = mx.nd.maximum(label.sum(axis=0), 1e-5)
            mean_pos = (cam_pos * label).sum(axis=0) / label_cnt
            mean_neg = (cam_neg * label).sum(axis=0) / label_cnt

            curr_score = mx.nd.concat(mean_pos.reshape(0, 1), mean_neg.reshape(0, 1), dim=1)
            empty_class = (label_cnt < 0.999).reshape(0, 1)
            curr_score = curr_score * (1 - empty_class) + accum_score * empty_class

            self.kvstore.push(0, curr_score)
            self.kvstore.pull(0, accum_score)

        sign = ((accum_score[:, 0] < accum_score[:, 1]) - 0.5) * (-2)
        new_score = score * sign.reshape(1, -1, 1, 1)
        self.assign(out_data[0], req[0], new_score)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for grad in in_grad:
            grad[:] = 0

@mx.operator.register('ICD_TopDown')
class ICD_TopDownProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1, warmup=0):
        super(ICD_TopDownProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.warmup = int(warmup)

    def list_arguments(self):
        return ['data', 'label', 'score']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        n, c, h, w = in_shape[0]
        _, C = in_shape[1]
        return in_shape, [[n, C, h, w]], []

    def create_operator(self, ctx, shapes, dtypes):
        return ICD_TopDown(self.grad_scale, self.warmup)

class ICD_TopDown(mx.operator.CustomOp):
    def __init__(self, grad_scale, warmup):
        self.grad_scale = grad_scale
        self.warmup = warmup
        self.max_warmup = max(warmup, 1)

    def get_warmup_scale(self):
        if self.warmup <= 0: return 1
        scale = np.exp(-10. * self.warmup * 1. / self.max_warmup)
        self.warmup -= 1
        return scale

    def forward(self, is_train, req, in_data, out_data, aux):
        data, label, score = in_data
        data_sigmoid = mx.nd.sigmoid(data)
        self.assign(out_data[0], req[0], data_sigmoid)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data, label, score = in_data
        data_sigmoid = out_data[0]
        n, c, h, w = data.shape

        score_norm = mx.nd.maximum(score, 0) / mx.nd.maximum(score.max(axis=(2, 3), keepdims=True), 1e-5) + \
                mx.nd.minimum(score, 0) / (-mx.nd.minimum(score.min(axis=(2, 3), keepdims=True), -1e-5))
        mask = mx.nd.abs(score_norm) > 0.2

        target = score_norm > 0
        grad = (data_sigmoid - target) * label.reshape(0, 0, 1, 1) * mask
        grad = grad * (self.grad_scale * self.get_warmup_scale() / (h * w))

        self.assign(in_grad[0], req[0], grad)
        in_grad[1][:] = 0
        in_grad[2][:] = 0

@mx.operator.register('SPRefine')
class SPRefineProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SPRefineProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'superpixel', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return SPRefine()

class SPRefine(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        data, superpixel, label = in_data
        n, c, h, w = data.shape
        _, h2, w2 = superpixel.shape

        sp_num = int(superpixel.max().asnumpy()) + 1
        sp_flat = mx.nd.one_hot(superpixel, depth=sp_num).transpose((0, 3, 1, 2))
        if h != h2:
            sp_flat = mx.nd.contrib.BilinearResize2D(sp_flat, height=h, width=w)
        sp_flat = sp_flat.reshape(0, 0, -3)

        score_flat = data.reshape(0, 0, -3) * label.reshape(0, 0, 1)
        score_sign = mx.nd.sign(score_flat)

        sp_sign = mx.nd.batch_dot(score_sign, sp_flat, transpose_b=True)
        sp_fg = sp_sign.max(axis=1, keepdims=True) > 1e-5
 
        sp_flat_avg = sp_flat / mx.nd.maximum(sp_flat.sum(axis=2, keepdims=True), 1e-5)
        score_flat_pos = mx.nd.maximum(score_flat, 0) / mx.nd.maximum(score_flat.max(axis=1, keepdims=True), 1e-5)
        sp_fg_score = mx.nd.batch_dot(score_flat_pos, sp_flat_avg, transpose_b=True)
        sp_fg_label = mx.nd.one_hot(sp_fg_score.argmax(axis=1), depth=c).transpose((0, 2, 1))
        sp_fg = sp_fg * sp_fg_label * label.reshape(0, 0, 1)
        sp_bg = 1 - sp_fg

        score_avg_fg = mx.nd.batch_dot(mx.nd.maximum(score_flat, 0), sp_flat_avg, transpose_b=True)
        score_avg_bg = mx.nd.batch_dot(mx.nd.minimum(score_flat, 0), sp_flat_avg, transpose_b=True)
        score_avg = (score_avg_fg * sp_fg) + (score_avg_bg * sp_bg)
        score_avg = mx.nd.batch_dot(score_avg, sp_flat).reshape(0, 0, h, w)

        self.assign(out_data[0], req[0], score_avg)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

