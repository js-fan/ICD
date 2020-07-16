import mxnet as mx
import numpy as np
import cv2
import os
from ...utils.dataset_tools import VOC as vocUtils
import multiprocessing as mp

class SuperpixelLoader(mx.io.DataIter):
    def __init__(self, image_root, annotation_root, superpixel_root,
            data_list, batch_size, target_size,
            pad_dataset=False, shuffle=False, rand_scale=False, rand_mirror=False, rand_crop=False):
        with open(data_list, 'r') as f:
            data_names = [x.strip() for x in f.readlines()]

        if pad_dataset and (len(data_names) % batch_size > 0):
            pad_num = batch_size - (len(data_names) % batch_size)
            data_names = data_names + data_names[:pad_num]

        self.image_src_list = [os.path.join(image_root, x+'.jpg') for x in data_names]
        self.label_list = [vocUtils.get_annotation(os.path.join(annotation_root, x+'.xml')) \
                for x in data_names] if annotation_root is not None else [None] * len(data_names)
        self.superpixel_src_list = [os.path.join(superpixel_root, x+'.png') for x in data_names]

        self.batch_size = batch_size
        self.target_size = target_size

        self.shuffle = shuffle
        self.rand_scale = rand_scale
        self.rand_mirror = rand_mirror
        self.rand_crop = rand_crop

        scale_pool = [0.5, 0.75, 1, 1.25, 1.5]
        self.scale_sampler = lambda : np.random.choice(scale_pool)
    
        self.index = list(range(len(data_names)))
        self.num_batch = len(data_names) // self.batch_size
        self.reset()

    def reset(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def next(self):
        if self.current >= self.num_batch:
            raise StopIteration
        index = self.index[self.current*self.batch_size : (self.current+1)*self.batch_size]
        self.current += 1

        image_src_list = [self.image_src_list[i] for i in index]
        label_list = [self.label_list[i] for i in index]
        superpixel_src_list = [self.superpixel_src_list[i] for i in index]
        self.cache_image_src_list = image_src_list

        batch = load_batch_sp(image_src_list, label_list, superpixel_src_list,
                self.target_size, self.scale_sampler, self.rand_scale, self.rand_mirror, self.rand_crop)
        return batch

def load_batch_sp(image_src_list, label_list, superpixel_src_list, target_size, scale_sampler, rand_scale,
        rand_mirror, rand_crop, num_class=20, pool=None):
    if pool is None:
        #print("single process:")
        img_batch, sp_batch = [], []
        for image_src, superpixel_src in zip(image_src_list, superpixel_src_list):
            img, sp = load_img_sp(image_src, superpixel_src, target_size, scale_sampler() if rand_scale else 1,
                    rand_mirror and (np.random.rand() > 0.5), rand_crop)
            img_batch.append(img)
            sp_batch.append(sp)
    else:
        #print("multi process:")
        scales = [scale_sampler() if rand_scale else 1 for _ in image_src_list]
        mirrors = [rand_mirror and (np.random.rand() > 0.5) for _ in image_src_list]
        jobs = [pool.apply_async(load_img_sp, (image_src, superpixel_src, target_size, scale, mirror, rand_crop)) for image_src, superpixel_src, scale, mirror in zip(image_src_list, superpixel_src_list, scales, mirrors)]
        res = [job.get() for job in jobs]
        img_batch = [x[0] for x in res]
        sp_batch = [x[1] for x in res]

    img_batch = mx.nd.array(np.array(img_batch)).transpose((0, 3, 1, 2))
    img_batch = img_batch[:, ::-1, :, :]
    sp_batch = mx.nd.array(sp_batch)

    if label_list[0] is None:
        lbl_batch = None
    else:
        lbl_batch = [mx.nd.one_hot(mx.nd.array(label), depth=num_class).max(axis=0, keepdims=True) for label in label_list]
        lbl_batch = [mx.nd.concat(*lbl_batch, dim=0)]

    batch = mx.io.DataBatch(data=[img_batch, sp_batch], label=lbl_batch)
    return batch

def load_img_sp(image_src, superpixel_src, target_size, scale, mirror, rand_crop):
    img = cv2.imread(image_src)
    h, w = img.shape[:2]

    if superpixel_src is None:
        sp = np.zeros((h, w, 3), np.uint8)
    else:
        sp = cv2.imread(superpixel_src)
    
    if scale != 1:
        h = int(h * scale + .5)
        w = int(w * scale + .5)
        img = cv2.resize(img, (w, h))
        sp = cv2.resize(sp, (w, h), interpolation=cv2.INTER_NEAREST)

    sp = sp.astype(np.int32)
    sp = sp[..., 0] + sp[..., 1] * 256 + sp[..., 2] * 65536
    
    if mirror:
        img = img[:, ::-1]
        sp = sp[:, ::-1]

    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        sp = cv2.copyMakeBorder(sp, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=int(sp.max())+1)
        h, w = img.shape[:2]

    if rand_crop:
        h_bgn = np.random.randint(0, h - target_size + 1)
        w_bgn = np.random.randint(0, w - target_size + 1)
    else:
        h_bgn = (h - target_size) // 2
        w_bgn = (w - target_size) // 2

    img = img[h_bgn:h_bgn+target_size, w_bgn:w_bgn+target_size]
    sp = sp[h_bgn:h_bgn+target_size, w_bgn:w_bgn+target_size]

    sp_vals = np.unique(sp)
    sp_lookup = np.zeros((sp_vals.max()+1,), np.int32)
    sp_lookup[sp_vals] = np.random.permutation(len(sp_vals))
    sp = sp_lookup[sp.ravel()].reshape(sp.shape)

    return img, sp

