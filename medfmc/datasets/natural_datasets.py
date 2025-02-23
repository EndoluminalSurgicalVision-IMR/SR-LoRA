# https://github.com/open-mmlab/mmpretrain/blob/master/mmcls/datasets/base_dataset.py
from mmcls.datasets.builder import DATASETS
from mmcls.datasets import CIFAR100
import os
import os.path
import pickle

import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmcls.datasets.utils import check_integrity, download_and_extract_archive


from .medical_datasets import BaseDataset
    
@DATASETS.register_module()
class CIFAR100_Fewshot(CIFAR100):
    def __init__(self, csv_path=None, **kwargs):
        self.csv_path = csv_path  # Store csv_path locally
        # Call the parent class initializer with valid arguments
        super(CIFAR100_Fewshot, self).__init__(**kwargs)

        print('**********Build {} CIFAR100-Few-shot ***********'.format(self.flag))
        print('******************** LEN ***********', len(self.data_infos))
        print('*********few shot************', (self.csv_path))
    
        
    def load_annotations(self):

        rank, world_size = get_dist_info()

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        if not self.test_mode:
            downloaded_list = self.train_list
            self.flag = 'train'
        else:
            downloaded_list = self.test_list
            self.flag = 'test'
        
        if self.csv_path is not None:
            if 'val' in self.csv_path:
                downloaded_list = self.train_list
                self.flag = 'val'

        self.imgs = []
        self.gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_prefix, self.base_folder,
                                     file_name)
            # filename ==  'train' /'test'
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
    
                ### selct few-shot sampels from the pre-defined .csv file
                selected_data = []
                selected_labels = []
                if self.csv_path is not None:
                    import pandas as pd
                    df = pd.read_csv(self.csv_path)
                    selected_filenames = df['file_name'].tolist()
                    selected_classes = df['class'].tolist()
                    filename_to_class = dict(zip(selected_filenames, selected_classes))
                    for idx, filename in enumerate(entry['filenames']):
                        if filename in selected_filenames:
                            selected_data.append(entry['data'][idx])
                            if 'labels' in entry:
                                selected_labels.append(entry['labels'][idx])
                            else:
                                selected_labels.append(entry['fine_labels'][idx])
                    self.imgs.extend(selected_data)
                    self.gt_labels.extend(selected_labels)
                    
                else:
                    self.imgs.append(entry['data'])
                    if 'labels' in entry:
                        self.gt_labels.extend(entry['labels'])
                    else:
                        self.gt_labels.extend(entry['fine_labels'])
                    


        self.imgs = np.vstack(self.imgs).reshape(-1, 3, 32, 32)
        self.imgs = self.imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []
        for img, gt_label in zip(self.imgs, self.gt_labels):
            gt_label = np.array(gt_label, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class VTAB1kCIFAR100(BaseDataset):

    CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
        'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
        'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
        'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]


    def __init__(self, **kwargs):
        super(VTAB1kCIFAR100, self).__init__(**kwargs)
        print('**********Build VTAB1kCIFAR100 ***********')
        print('******************** LEN ***********', len(self.data_infos))
        print(self.data_infos[0])

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
              
                # k = item.find('.jpg')
                # filename = item[:k+4]
                # imglabel = int(item[-1:])
                filename, imglabel = item.split(' ')
                print('*******finename**** {}'.format(filename))
                imglabel = int(imglabel)
                gt_label = np.array(imglabel, dtype=np.int64)

                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos
    
    
@DATASETS.register_module()
class VTAB1k(BaseDataset):

    def __init__(self, **kwargs):
        super(VTAB1k, self).__init__(**kwargs)
        print('**********Build VTAB1k ***********')
        print('******************** LEN ***********', len(self.data_infos))
        print(self.data_infos[0])

    def load_annotations(self):

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip() for x in f.readlines()]
            for item in samples:
                filename, imglabel = item.split()
                imglabel = int(imglabel)
                
                gt_label = np.array(imglabel, dtype=np.int64)
                # print('*******finename**** {} *******gtlabel****** {}'.format(filename, gt_label))


                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = gt_label

                data_infos.append(info)

        return data_infos
