import multiprocessing as mp
import numpy as np
import cv2
import os

def compute_iou(names, num_cls, target_root, gt_root, num_threads=16, arr_=None):
    _compute_iou = lambda x: np.diag(x) / (x.sum(axis=0) + x.sum(axis=1) - np.diag(x) + 1e-10)
    if isinstance(names, str):
        with open(names) as f:
            names = [name.strip() for name in f.readlines()]

    if num_threads == 1:
        mat = np.zeros((num_cls, num_cls), np.float32)
        for name in names:
            gt = cv2.imread(os.path.join(gt_root, name+'.png'), 0).astype(np.int32)
            pred = cv2.imread(os.path.join(target_root, name+'.png'), 0).astype(np.int32)
            if gt.shape != pred.shape:
                info(None, "Name {}, gt.shape != pred.shape: [{} vs. {}]".format(name, gt.shape, pred.shape))
                continue
            valid = (gt < num_cls) & (pred < num_cls)
            mat += np.bincount(gt[valid] * num_cls + pred[valid], minlength=num_cls**2).reshape(num_cls, -1)

        if arr_ is not None:
            arr_mat = np.frombuffer(arr_.get_obj(), np.float32)
            arr_mat += mat.ravel()
        else:
            return _compute_iou(mat.copy())
    else:
        workload = np.full((num_threads,), len(names)//num_threads, np.int32)
        if workload.sum() < len(names):
            workload[:(len(names) - workload.sum())] += 1
        workload = np.cumsum(np.hstack([0, workload]))
        names_split = [names[i:j] for i, j in zip(workload[:-1], workload[1:])]

        arr_ = mp.Array('f', np.zeros((num_cls * num_cls,), np.float32))
        mat = np.frombuffer(arr_.get_obj(), np.float32).reshape(num_cls, -1)
        jobs = [mp.Process(target=compute_iou, args=(_names, num_cls, target_root, gt_root, 1, arr_))
                for _names in names_split]
        [job.start() for job in jobs]
        [job.join() for job in jobs]
        return _compute_iou(mat.copy())

if __name__ == '__main__':
    gt = 'VOC2012/extra/SegmentationClassAug'
    target = 'snapshot/icd/results/seeds/crf'
    iou = compute_iou('./data/VOC2012/train_aug.txt', 21, target, gt)
    print('[{}] mIoU: {}\n{}'.format(target, iou.mean(), iou))

