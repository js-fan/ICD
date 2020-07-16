import argparse
from core.data.VOC import SuperpixelLoader
from core.utils import *
from core.model.vgg import *
from core.model.layers_custom import *
import multiprocessing as mp
import pydensecrf.densecrf as dcrf

def norm_score(score):
    score = score - score.min()
    score /= max(score.max(), 1e-5)
    return score

def get_score_map(score, TYPE=cv2.COLORMAP_JET):
    score = norm_score(score)
    score_cmap = cv2.applyColorMap((score * 255.99).astype(np.uint8), TYPE)
    return score_cmap

def build_model(args, for_training=True):
    x_img = mx.sym.Variable('data') - 127
    x_lbl = mx.sym.Variable('label')
    x_sp = mx.sym.Variable('superpixel')

    # CAM branch
    x_cam = vgg16_cam(x_img, num_cls=args.num_cls)
    x_cam_flatten = Flatten(Pool(x_cam, kernel=(1, 1), global_pool=True, pool_type='avg'))
    x_cam_sigmoid = mx.sym.Custom(x_cam_flatten, x_lbl, op_type='MultiSigmoidLoss')

    # ICD branch
    # buttom up
    x_feat = x_cam.get_internals()['conv5_3_relu_output']
    x_feat = BN(x_feat, fix_gamma=True, use_global_stats=not for_training, name='feat_bn')

    x_icd_bu = mx.sym.Custom(x_feat, x_lbl, grad_scale=10, op_type='ICD_BottomUp', name='icd_bu')
    x_icd_bu = mx.sym.Custom(x_icd_bu, x_cam, x_lbl, momentum=0.9, op_type='AccumResign', name='icd_accum_resign')

    # top down
    x_feat1 = Relu(Conv(x_feat, num_filter=1024, kernel=(3, 3), pad=(1, 1), name='fc6_ft'))
    x_feat2 = Relu(Conv(x_feat1, num_filter=1024, kernel=(1 ,1), name='fc7_ft'))
    x_feat = mx.sym.concat(x_feat, x_feat1, x_feat2, dim=1)

    x_icd_bu_sp = mx.sym.Custom(x_icd_bu, x_sp, x_lbl, op_type='SPRefine')
    x_icd_td = Conv(x_feat, num_filter=args.num_cls, kernel=(1, 1), no_bias=True, name='conv_ft', lr_mult=10)
    x_icd_td_ = mx.sym.Custom(x_icd_td, x_lbl, x_icd_bu_sp, warmup=2*args.num_sample//args.batch_size, op_type='ICD_TopDown', name='icd_td')

    #
    if for_training:
        symbol = mx.sym.Group([x_cam_sigmoid, x_icd_bu, x_icd_td_])
    else:
        symbol = mx.sym.Group([x_cam, x_icd_bu, x_icd_bu_sp, x_icd_td])

    mod = mx.mod.Module(symbol, data_names=('data', 'superpixel'), label_names=('label',),
            context=[mx.gpu(int(gpu_id)) for gpu_id in setGPU(args.gpus).split(',')])
    mod.bind(data_shapes=[('data', (args.batch_size, 3, args.image_size, args.image_size)),
        ('superpixel', (args.batch_size, args.image_size, args.image_size))],
        label_shapes=[('label', (args.batch_size, args.num_cls))])

    if for_training:
        pretrained = args.pretrained
    else:
        pretrained = os.path.join(args.snapshot, '%s-%04d.params' % (args.model, args.num_epoch-1))
        if not os.path.exists(pretrained):
            pretrained = args.pretrained
    info(None, "Using pretrained params: {}".format(pretrained), 'red')

    arg_params, aux_params = loadParams(pretrained)
    if for_training:
        for name in ['fc6', 'fc7']:
            arg_params[name + '_ft_weight'] = arg_params[name + '_weight'].copy()
            arg_params[name + '_ft_bias'] = arg_params[name + '_bias'].copy()
            info(None, 'Initialized %s_ft by %s' % (name, name), 'red')
    arg_params, aux_params = checkParams(mod, arg_params, aux_params, initializer=mx.init.Normal(0.01))
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    if for_training:
        mod.init_optimizer(optimizer='sgd', optimizer_params={
            'learning_rate': args.learning_rate,
            'momentum': 0.9,
            'wd': 5e-4})
    return mod

def run_training(args):
    loader = SuperpixelLoader(args.image_root, args.annotation_root, args.superpixel_root,
            args.data_list, args.batch_size, args.image_size,
            shuffle=True, rand_scale=True, rand_mirror=True, rand_crop=True)
    args.num_sample = len(loader.image_src_list)

    mod = build_model(args, for_training=True)

    saveParams = SaveParams(mod, args.snapshot, args.model, args.num_save)
    lrScheduler = LrScheduler('poly', args.learning_rate, {'num_epoch': args.num_epoch, 'power': 0.9})

    logger = getLogger(args.snapshot, args.model)
    summaryArgs(logger, vars(args), 'green')

    for n_epoch in range(args.begin_epoch, args.num_epoch):
        loader.reset()
        icd_bu_max, icd_bu_min = 0, 0
        cls_loss, cls_correct, cls_total = 0, 0, 0

        mod._optimizer.lr = lrScheduler.get(n_epoch)
        info(logger, "Learning rate: {}".format(mod._optimizer.lr), 'yellow')

        Timer.record()
        v_images = []
        for n_batch, batch in enumerate(loader, 1):
            mod.forward_backward(batch)
            mod.update()

            # monitor
            if n_batch % args.log_frequency == 0:
                outputs = [x.asnumpy() for x in mod.get_outputs()]
                cam_sigmoid, icd_bu, icd_td = outputs
                n, c, h, w = icd_bu.shape
                label = batch.label[0].asnumpy()
                Ls_list = [np.nonzero(_lbl)[0] for _lbl in label]

                loss_mom = (float(n_batch) - args.log_frequency) / n_batch
                lossUpdate = lambda old, new: old * loss_mom + float(new) * (1 - loss_mom)

                icd_bu_max = lossUpdate(icd_bu_max, icd_bu.max())
                icd_bu_min = lossUpdate(icd_bu_min, icd_bu.min())

                _cls_loss = -np.log(np.maximum(cam_sigmoid, 1e-5))*label-np.log(np.maximum(1-cam_sigmoid, 1e-5))*(1-label)
                cls_loss = lossUpdate(cls_loss, _cls_loss.sum(axis=1).mean())

                for probs_, label_ in zip(cam_sigmoid, Ls_list):
                    probs_rank = (-probs_).argsort()
                    for i in range(len(label_)):
                        if int(probs_rank[i]) in label_:
                            cls_correct += 1
                    cls_total += len(label_)
                cls_rankacc = cls_correct / cls_total

                # visualize some results
                vis_idx_list = []
                for i, Ls in enumerate(Ls_list):
                    if len(Ls) <= 2:
                        vis_idx_list.append((i, Ls))
                    if len(vis_idx_list) >= 2:
                        break
                for i, Ls in vis_idx_list:
                    image = batch.data[0][i].asnumpy().transpose(1, 2, 0).astype(np.uint8)[..., ::-1]
                    getScoreMap = lambda x: cv2.addWeighted(get_score_map(cv2.resize(x, image.shape[:2])), 0.8, image, 0.2, 0)
                    for L in Ls:
                        h_images = [image]
                        icd_bu_ = icd_bu[i][L]
                        h_images.append(getScoreMap(np.maximum(icd_bu_, 0)))
                        h_images.append(getScoreMap(np.maximum(-icd_bu_, 0)))
                        icd_td_ = -np.log(np.maximum(1./(icd_td[i][L]+1e-5) - 1, 1e-5))
                        h_images.append(getScoreMap(np.maximum(icd_td_, 0)))
                        h_images.append(getScoreMap(np.maximum(-icd_td_, 0)))
                        v_images.append(imhstack(h_images, height=120))

                # log
                Timer.record()
                msg = 'Epoch={}, Batch={}, icdMin={:.3f}, icdMax={:.3f}, clsAcc={:.3f}, clsLoss={:.3f}, speed={:.1f} b/s'
                msg = msg.format(n_epoch, n_batch, icd_bu_min, icd_bu_max, cls_rankacc, cls_loss,
                        args.log_frequency/Timer.interval())
                info(logger, msg)

        v_images = imvstack(v_images)
        imwrite(os.path.join(args.snapshot, args.model+'_visdemo', '%s-%04d.jpg' % (args.model, n_epoch)), v_images)

        saved_params = saveParams(n_epoch)
        info(logger, "Saved checkpoint:" + "\n  ".join(saved_params), 'green')

def run_infer(args):
    mod = build_model(args, for_training=False)
    loader = SuperpixelLoader(args.image_root, args.annotation_root, args.superpixel_root,
            args.data_list, args.batch_size, args.image_size,
            pad_dataset=True, shuffle=False, rand_scale=False, rand_mirror=False, rand_crop=False)

    # visualize some demos
    v_images = [[] for _ in range(args.num_cls)]
    NUM = 20

    for n_batch, batch in enumerate(loader, 1):
        mod.forward(batch, is_train=False)
        outputs = [x.asnumpy() for x in mod.get_outputs()]
        cam, icd_bu, icd_bu_sp, icd_td = outputs

        image_src_list = loader.cache_image_src_list
        label = batch.label[0].asnumpy()
        N, C, H, W = icd_td.shape

        for img_src, label, cam, icd_bu, icd_bu_sp, icd_td in zip(image_src_list, label, cam, icd_bu, icd_bu_sp, icd_td):
            Ls = np.nonzero(label)[0]
            cam = cam[Ls]
            icd = icd_td[Ls]

            name = os.path.basename(img_src).rsplit('.', 1)[0]
            npsave(os.path.join(args.snapshot, 'results', 'scores_cam', name+'.npy'), cam)
            npsave(os.path.join(args.snapshot, 'results', 'scores_icd', name+'.npy'), icd)

            # demo
            if len(Ls) == 1:
                L = Ls[0]
                if len(v_images[L]) < NUM:
                    image = cv2.imread(img_src)
                    h, w = image.shape[:2]
                    getScoreMap = lambda x: cv2.addWeighted(get_score_map(cv2.resize(x, (args.image_size,)*2)[:h, :w]),
                        0.8, image, 0.2, 0)
                    visScores = lambda x: [getScoreMap(np.maximum(x, 0)), getScoreMap(np.maximum(-x, 0))]
                    h_images = sum([[image]] + list(map(visScores, [cam[0], icd_bu[L], icd_bu_sp[L], icd_td[0]])), [])
                    v_images[L].append(imhstack(h_images, height=120))
                elif len(v_images[L]) == NUM:
                    img = imvstack(v_images[L])
                    imwrite(os.path.join(args.snapshot, 'results', 'scores_demo', 'class_%d.jpg' % L), img)
                    v_images[L].append(None)

    for L, v_images in enumerate(v_images):
        if v_images and v_images[-1] is not None:
            img = imvstack(v_images)
            imwrite(os.path.join(args.snapshot, 'results', 'scores_demo', 'class_%d.jpg' % L), img)

def _generate_seed(name, cam_root, icd_root, img_root, ann_root, save_root, infer_size, num_cls, confidence):
    label = VOC.get_annotation(os.path.join(ann_root, name+'.xml'))

    image = cv2.imread(os.path.join(img_root, name+'.jpg'))
    h, w = image.shape[:2]

    scores = np.load(os.path.join(icd_root, name+'.npy'))
    scores = np.array([cv2.resize(x, (infer_size, infer_size))[:h, :w] for x in scores])

    scores_fg = np.maximum(scores, 0) / np.maximum(scores.max(axis=(1, 2), keepdims=True), 1e-5)
    proposal_fg = scores_fg > 1e-5
    candidate_fg = proposal_fg.argmax(axis=0)

    if len(label) > 1:
        cams = np.load(os.path.join(cam_root, name+'.npy'))
        cams = np.array([cv2.resize(x, (infer_size, infer_size))[:h, :w] for x in cams])
        comp_scores = norm_score(np.maximum(cams, 0)) * scores_fg
        conflict_fg = proposal_fg.sum(axis=0) > 1
        candidate_fg = candidate_fg * (1 - conflict_fg) + comp_scores.argmax(axis=0) * conflict_fg

    fg = proposal_fg.max(axis=0)

    scores_bg = np.minimum(scores, 0) / np.minimum(scores.min(axis=(1, 2), keepdims=True), -1e-5)
    bg = (scores_bg > 1e-5).min(axis=0)

    undefined = ~(bg ^ fg)

    if len(label) > 1:
        undefined |= ((comp_scores.max(axis=0) - comp_scores.min(axis=0)) <= 1e-5) & fg

    seed = (np.array(label) + 1)[candidate_fg.ravel()].reshape(h, w).astype(np.uint8)
    seed[bg] = 0
    seed[undefined] = 255
    imwrite(os.path.join(save_root, 'nocrf', name+'.png'), seed)

    # crf refine
    seed_prob = confidence
    res_prob = (1 - seed_prob) / num_cls

    prob = np.full((256, h*w), res_prob, np.float32)
    prob[seed.ravel(), np.arange(h*w)] = seed_prob
    prob = prob[:num_cls]
    prob /= prob.sum(axis=0, keepdims=True)
    u = -np.log(np.maximum(prob, 1e-5))

    d = dcrf.DenseCRF2D(w, h, num_cls)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image[..., ::-1].copy(), compat=10)

    prob_crf = d.inference(10)
    prob_crf = np.array(prob_crf).reshape(num_cls, h, w)
    prob_crf /= prob_crf.sum(axis=0, keepdims=True)

    seed_crf = prob_crf.argmax(axis=0).astype(np.uint8)
    seed_crf[prob_crf.max(axis=0) < 0.1] = 255
    imwrite(os.path.join(save_root, 'crf', name+'.png'), seed_crf)

def run_generate_seeds(args, num_process=32):
    cam_root = os.path.join(args.snapshot, 'results', 'scores_cam')
    icd_root = os.path.join(args.snapshot, 'results', 'scores_icd')
    img_root = args.image_root
    ann_root = args.annotation_root
    infer_size = args.image_size
    save_root = os.path.join(args.snapshot, 'results', 'seeds')
    num_cls = args.num_cls + 1 # includes bg
    confidence = 0.5

    with open(args.data_list) as f:
        names = [x.strip() for x in f.readlines()]

    pool = mp.Pool(num_process)
    jobs = [pool.apply_async(_generate_seed, (name, cam_root, icd_root, img_root, ann_root, save_root,
        infer_size, num_cls, confidence)) for name in names]
    [job.get() for job in jobs]


if __name__ == '__main__':
    dataset_root = 'Your-VOC2012'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, default=dataset_root+'/JPEGImages')
    parser.add_argument('--annotation-root', type=str, default=dataset_root+'/Annotations')
    parser.add_argument('--superpixel-root', type=str, default='./data/superpixels/voc_superpixels')
    parser.add_argument('--data-list',  type=str, default='./data/VOC2012/train_aug.txt')

    parser.add_argument('--model',      type=str, default='vgg16_icd')
    parser.add_argument('--pretrained', type=str, default='./data/pretrained/vgg16_20M.params')
    parser.add_argument('--num-cls',    type=int, default=20)

    parser.add_argument('--image-size',     type=int, default=321)
    parser.add_argument('--batch-size',     type=int, default=32)
    parser.add_argument('--learning-rate',  type=float, default=1e-3)
    parser.add_argument('--num-epoch',      type=int, default=20)
    parser.add_argument('--begin-epoch',    type=int, default=0)
    parser.add_argument('--num-sample',     type=int, default=0)

    parser.add_argument('--snapshot', type=str, default='./snapshot/icd')
    parser.add_argument('--num-save', type=int, default=5)
    parser.add_argument('--log-frequency', type=int, default=5)
    parser.add_argument('--gpus', type=str, default='0,1')

    args = parser.parse_args()
    args.log_frequency = max(10582 // (args.batch_size * 20), 5)

    assert os.path.exists(args.image_root)
    assert os.path.exists(args.annotation_root)
    assert os.path.exists(args.superpixel_root)
    assert os.path.exists(args.pretrained)

    # train
    run_training(args)

    # infer icd scores
    args.batch_size = 4
    args.image_size = 513
    run_infer(args)

    # generate seeds
    run_generate_seeds(args)

