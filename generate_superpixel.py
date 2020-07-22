import argparse
from core.model.vgg import vgg16_cam
from core.utils import *
from core.data.VOC import ClassLoader
import multiprocessing as mp
from subprocess import call

def extract_dense_features(args):
    # model
    x = mx.sym.Variable('data')
    x = vgg16_cam(x, 1).get_internals()['conv5_3_relu_output']
    mod = mx.mod.Module(x, data_names=('data',), label_names=None,
            context=[mx.gpu(int(gpu_id)) for gpu_id in args.gpus.split(',')],
            fixed_param_names=None)
    mod.bind(data_shapes=[('data', (args.batch_size, 3, args.image_size, args.image_size))],
            for_training=False)

    info(None, "Using pretrained params: {}".format(args.pretrained))
    assert os.path.exists(args.pretrained), args.pretrained
    arg_params, aux_params = loadParams(args.pretrained)
    arg_params, aux_params = checkParams(mod, arg_params, aux_params)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # data 
    loader = ClassLoader(args.image_root, None, args.data_list, args.batch_size,
            args.image_size, pad_dataset=True)

    # extract & save features
    save_root = os.path.join(args.save_feature_root, 'dense_feature')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for batch in loader:
        mod.forward(batch, is_train=False)
        features = mod.get_outputs()[0].asnumpy()

        names = [os.path.basename(src).rsplit('.', 1)[0] for src in loader.cache_image_src_list]
        for name, feature in zip(names, features):
            np.save(os.path.join(save_root, name+'.npy'), feature)

def _extract_superpixel_features(name, image_dir, sp_dir, dense_feat_dir, sp_feat_dir, infer_size, delete_dense_feature=False, gpu=None):
    image = cv2.imread(os.path.join(image_dir, name+'.jpg'))
    h, w = image.shape[:2]
    sp = cv2.imread(os.path.join(sp_dir, name+'.png')).astype(np.int32)
    sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
    assert sp.shape == (h, w), (name, image.shape, sp.shape)

    context = mx.cpu() if gpu is None else mx.gpu(gpu)
    feature = np.load(os.path.join(dense_feat_dir, name+'.npy'))
    feature = mx.nd.array(feature[np.newaxis], ctx=context)
    feature = mx.nd.contrib.BilinearResize2D(feature, height=infer_size, width=infer_size)
    feature = feature[0, :, :h, :w].reshape(-1, h * w)

    sp = mx.nd.one_hot(mx.nd.array(sp, ctx=context), depth=sp.max()+1).reshape(h * w, -1)
    sp_feature = mx.nd.dot(sp, feature, transpose_a=True, transpose_b=True)
    sp_feature /= mx.nd.maximum(sp.sum(axis=0, keepdims=True).T, 1e-5)
    sp_feature = sp_feature.asnumpy()
    np.save(os.path.join(sp_feat_dir, name+'.npy'), sp_feature)

    if delete_dense_feature:
        call(['rm', os.path.join(dense_feat_dir, name+'.npy')])

def extract_superpixel_features(args, num_threads=16):
    dense_feature_root = os.path.join(args.save_feature_root, 'dense_feature')
    save_root = os.path.join(args.save_feature_root, 'sp_feature')
    assert os.path.exists(dense_feature_root), dense_feature_root
    assert os.path.exists(args.image_root), args.image_root
    assert os.path.exists(args.superpixel_root), args.superpixel_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    with open(args.data_list) as f:
        names = [x.strip() for x in f.readlines()]
    assert len(os.listdir(dense_feature_root)) >= len(names)
    assert len(os.listdir(args.superpixel_root)) >= len(names)
    assert len(os.listdir(args.image_root)) >= len(names)

    pool = mp.Pool(num_threads)
    jobs = [pool.apply_async(_extract_superpixel_features, args=(name, args.image_root, args.superpixel_root,
        dense_feature_root, save_root, args.image_size, False, None)) for name in names]
    [job.get() for job in jobs]
    
def recalibrate_superpixel_index(sp):
    keys = np.unique(sp)
    vals = np.arange(len(keys))
    lookup = np.zeros((keys.max()+1,), np.int32)
    lookup[keys] = vals
    sp_new = lookup[sp.ravel()].reshape(sp.shape)
    return sp_new

def _greedy_merge_superpixels(name, feat_dir, sp_dir, save_dir, num):
    sp = cv2.imread(os.path.join(sp_dir, name+'.png')).astype(np.int32)
    sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
    h, w = sp.shape
    sp_onehot = np.zeros((sp.max() + 1, h * w), np.int32)
    sp_onehot[sp.ravel(), np.arange(h*w)] = 1
    sp_size = sp_onehot.sum(axis=1)

    feature = np.load(os.path.join(feat_dir, name+'.npy'))

    func_sim_feature = lambda i, j: feature[i].dot(feature[j]) / \
            ( np.linalg.norm(feature[i]) * np.linalg.norm(feature[j]) )
    func_sim_size = lambda i, j : 1. - (sp_size[i] + sp_size[j]) / (h * w)
    compute_sim = lambda i, j: func_sim_feature(i, j) + func_sim_size(i, j)

    # init sim mat
    sp_index = sp_onehot.max(axis=1)
    sim_matrix = np.zeros((sp_index.size, sp_index.size), np.float32)

    for i in range(sp_index.size):
        sim_matrix[i, i] = -np.inf
        for j in range(i+1, sp_index.size):
            sim_matrix[i, j] = compute_sim(i, j)
            sim_matrix[j, i] = sim_matrix[i, j]

    # greedy merge
    while sp_index.sum() > num:
        i, j = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
        if j < i:
            i, j = j, i

        sp_onehot[i] = np.maximum(sp_onehot[i], sp_onehot[j])
        sp_onehot[j, :] = 0
        feature[i] = feature[i] * (sp_size[i] / (sp_size[i] + sp_size[j])) + \
                feature[j] * (sp_size[j] / (sp_size[i] + sp_size[j]))
        sp_size[i] = sp_size[i]  + sp_size[j]

        sp_index = sp_onehot.max(axis=1)
        for k in range(sp_index.size):
            sim_matrix[j, k] = -np.inf
            sim_matrix[k, j] = -np.inf

            if sp_index[k] > 0 and (k != i):
                sim_matrix[i, k] = compute_sim(i, k)
                sim_matrix[k, i] = sim_matrix[i, k]

    merge_sp = sp_onehot.argmax(axis=0)
    merge_sp = recalibrate_superpixel_index(merge_sp)
    sp_8uc3 = np.array([merge_sp, np.zeros_like(merge_sp), np.zeros_like(merge_sp)])
    sp_8uc3 = sp_8uc3.astype(np.uint8).T.reshape(h, w, 3)
    cv2.imwrite(os.path.join(save_dir, name+'.png'), sp_8uc3)

def greedy_merge_superpixels(args, num_threads=32):
    feature_root = os.path.join(args.save_feature_root, 'sp_feature')
    save_root = args.save_superpixel_root
    assert os.path.exists(feature_root), feature_root
    assert os.path.exists(args.superpixel_root), args.superpixel_root
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(args.data_list) as f:
        names = [x.strip() for x in f.readlines()]

    pool = mp.Pool(num_threads)
    jobs = [pool.apply_async(_greedy_merge_superpixels, (name, feature_root, args.superpixel_root, save_root, args.max_superpixel_num)) for name in names]
    [job.get() for job in jobs]

def demo(args, num, savename):
    palette = np.random.randint(0, 256, (1000, 3)).astype(np.uint8)
    with open(args.data_list) as f:
        names = [x.strip() for x in f.readlines()]

    demo_images = []
    for name in names[:num]:
        img = cv2.imread(os.path.join(args.image_root, name+'.jpg'))
        sp0 = cv2.imread(os.path.join(args.superpixel_root, name+'.png')).astype(np.int32)
        sp1 = cv2.imread(os.path.join(args.save_superpixel_root, name+'.png')).astype(np.int32)
        sp0 = sp0[..., 0] + sp0[..., 1] * 256 + sp0[..., 2] * 65536
        sp1 = sp1[..., 0] + sp1[..., 1] * 256 + sp1[..., 2] * 65536
        sp0 = palette[sp0.ravel()].reshape(img.shape)
        sp1 = palette[sp1.ravel()].reshape(img.shape)
        demo_images.append(imhstack([img, sp0, sp1], height=240))
    demo_images = imvstack(demo_images)
    imwrite(savename, demo_images)

if __name__ == '__main__':
    dataset_root = 'Your-VOC2012'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', type=str, default=dataset_root+'/JPEGImages')
    parser.add_argument('--superpixel-root', type=str, default='./external/superpixel/data/VOC')
    parser.add_argument('--data-list',  type=str, default='./data/VOC2012/train_aug.txt')

    parser.add_argument('--pretrained', type=str, default='./data/pretrained/vgg16_20M.params')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=513)
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--save-feature-root', type=str, default='./data/cache/features')
    parser.add_argument('--save-superpixel-root', type=str, default='./data/superpixels/voc_superpixels')
    parser.add_argument('--max-superpixel-num', type=int, default=64)

    args = parser.parse_args()

    extract_dense_features(args) 
    extract_superpixel_features(args)
    greedy_merge_superpixels(args)

    demo(args, 20, './superpixels_demo.jpg')

