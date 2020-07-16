import numpy as np
import xml.etree.ElementTree as ET

class _VOC_proto(object):
    @staticmethod
    def _get_palette():
        def bitget(bit, idx):
            return (bit & (1 << idx)) > 0
        cmap = []
        for i in range(256):
            r, g, b = 0, 0, 0
            idx = i
            for j in range(8):
                r = r | (bitget(idx, 0) << (7 - j))
                g = g | (bitget(idx, 1) << (7 - j))
                b = b | (bitget(idx, 2) << (7 - j))
                idx = idx >> 3
            cmap.append((b, g, r))
        return np.array(cmap).astype(np.uint8)

    def __init__(self):
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.palette = self._get_palette()

    def name2index(self, name):
        return self.categories.index(name)

    def index2name(self, index):
        return self.categories[index]

    def get_annotation(self, filename, use_diff=False):
        tree = ET.parse(filename)
        root = tree.getroot()
        annotation = []
        tmp_annotation = []
        for obj in root.findall('object'):
            cat = obj.find('name').text
            non_diff = 1 - int(obj.find('difficult').text)
            if use_diff or non_diff:
                annotation.append(self.name2index(cat))
            else:
                tmp_annotation.append(self.name2index(cat))
        annotation = list(set(annotation))
    
        if len(annotation) == 0:
            annotation += list(set(tmp_annotation))
        annotation.sort()
        return annotation

VOC = _VOC_proto()

'''
class _Cityscape(object):
    def __init__(self):
        from .cityscape_labels import labels
        self.ids = [L.id for L in labels if not L.ignoreInEval]
        self.id2label = {self.ids[i] : i for i in range(len(self.ids))}
        self.label2id = {i : self.ids[i] for i in range(len(self.ids))}
        self.palette = np.array([L.color for L in labels if not L.ignoreInEval]).astype(np.uint8)[..., ::-1]

Cityscape = _Cityscape()
'''


