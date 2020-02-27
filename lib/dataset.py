import pickle as pkl
import os.path as osp
import itertools

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


DATA_POST_FIX = {
    'acid': '_2',
    'iodine': '_3'
}

image_label_map = {
    0: 1,  # norm or lsil
    1: 0  # hsil
}


def regist_cervix_dataset(cfg):
    img_dir = cfg.DATASETS.CERVIX_IMG_DIR
    split_paths = [cfg.DATASETS.CERVIX_SPLIT_PATH_TRAIN,
                   cfg.DATASETS.CERVIX_SPLIT_PATH_VALID,
                   cfg.DATASETS.CERVIX_SPLIT_PATH_TEST]
    anno_path = cfg.DATASETS.CERVIX_ANNO_PATH
    data_type = cfg.DATASETS.CERVIX_DATA_TYPE
    if cfg.DATASETS.CERVIX_LABEL_TYPE == 'sil':
        label_map = {
            1: 0,  # lsil
            2: 1  # hsil
            # 2  background, reserved
        }
        thing_classes = ['lsil', 'hsil']
    elif cfg.DATASETS.CERVIX_LABEL_TYPE == 'hsil':
        label_map = {
            2: 0  # hsil
            # 1  background, reserved
        }
        thing_classes = ['hsil']
    else:
        print('Unexpected value in cfg.DATASETS.CERVIX_LABEL_TYPE', cfg.DATASETS.CERVIX_LABEL_TYPE)
        raise ValueError

    for d, split_path in zip(['train', 'valid', 'test'], split_paths):
        DatasetCatalog.register('cervix_' + d, lambda x=split_path: get_cervix_dicts(img_dir=img_dir,
                                                                                     split_path=x,
                                                                                     anno_path=anno_path,
                                                                                     data_type=data_type,
                                                                                     label_map=label_map
                                                                                     ))
        MetadataCatalog.get('cervix_' + d).set(thing_classes=thing_classes)
        MetadataCatalog.get('cervix_' + d).set(evaluator_type='')


def get_cervix_dicts(img_dir, split_path, anno_path, data_type, label_map):
    post_fix = DATA_POST_FIX[data_type.lower()]
    imgs_info = pkl.load(open(anno_path, 'rb'))
    with open(split_path, 'r') as f:
        names = sorted([x.strip() for x in f.readlines()])

    dataset_dicts = []
    for idx, name in enumerate(names):
        record = {}
        name += post_fix
        img_info = imgs_info[name]
        img_path = osp.join(img_dir, name + '.jpg')
        h, w = img_info['shape']

        record['file_name'] = img_path
        record['image_id'] = idx
        record['height'] = h
        record['width'] = w
        record['image_label'] = image_label_map[img_info['image_label']]

        img_annos = img_info['annos']
        if img_annos != [0]:
            objs = []
            for anno in img_annos:
                label = anno['label']
                if label not in label_map:  # Inflammation or lsil
                    continue
                label = label_map[label]
                poly = [int(x) for x in itertools.chain.from_iterable(anno['segm'])]

                obj = {
                    'bbox': anno['bbox'],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [poly],
                    'category_id': label,
                    'iscrowd': 0
                }
                objs.append(obj)
            record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts
