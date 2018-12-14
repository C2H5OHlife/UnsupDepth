import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/image_02/0000000.jpg
        root/scene_1/image_03/0000000.jpg
        root/scene_1/image_02/0000001.jpg
        root/scene_1/image_03/0000001.jpg
        ..
        root/scene_1/image_02/cam.txt
        root/scene_1/image_03/cam.txt
        root/scene_2/image_02/0000000.jpg
        root/scene_2/image_03/0000000.jpg

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            scene1 = scene / 'image_02'
            scene2 = scene / 'image_03'
            # left camera
            intrinsics = np.genfromtxt(scene1/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene1.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
            # right camera
            intrinsics = np.genfromtxt(scene2 / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene2.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)


class StereoSequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/image_02/0000000.jpg
        root/scene_1/image_03/0000000.jpg
        root/scene_1/image_02/0000001.jpg
        root/scene_1/image_03/0000001.jpg
        ..
        root/scene_1/image_02/cam.txt
        root/scene_1/image_03/cam.txt
        root/scene_2/image_02/0000000.jpg
        root/scene_2/image_03/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    # 爬虫
    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)  # [0 2]
        for scene in self.scenes:
            scene1 = scene/'image_02' #
            scene2 = scene/'image_03' #
            #for fold in open(list_path): #
            intrinsics = np.genfromtxt(scene1/'cam.txt').astype(np.float32).reshape((3, 3))  # 相机参数吗，给定的
            imgs1 = sorted(scene1.files('*.jpg'))  # 获取某个序列里所有图片
            imgs2 = sorted(scene2.files('*.jpg')) #
            if len(imgs1) < sequence_length:
                continue
            for i in range(demi_length, len(imgs1)-demi_length):
                #sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                sample = {'intrinsics': intrinsics, 'tgt': imgs1[i], 'ref_imgs': [], 'paral_img': imgs2[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs1[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        par_img = load_as_float(sample['paral_img'])
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs + [par_img], np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:-1]
            par_img = imgs[-1]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, par_img, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        # 一个序列是一个scene
        # 连着三帧的组合是一个sample
        return len(self.samples)
