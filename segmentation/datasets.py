import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data

from transform import HorizontalFlip, VerticalFlip


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def default_loader(path):
    return Image.open(path)




class BreastDataSet(data.Dataset):
    def __init__(self, root ='/home/projet16/sources/MCADAForPathonet/segmentation', 
                 split="train", img_transform=None, label_transform=None, test=True,
                 label_type=None, input_ch=3):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        if split == 'train':
            self.img_path_list_file = os.path.join(root, 'train_breast_tumor.txt')
            self.mask_path_list_file = os.path.join(root, 'train_breast_tumor_mask.txt')
        else:
            self.img_path_list_file = os.path.join(root, 'test_breast_tumor.txt')
            self.mask_path_list_file = os.path.join(root,'test_breast_tumor_mask.txt')
        with open(self.img_path_list_file, 'r') as f:
            content =  f.readlines()
        with open(self.mask_path_list_file, 'r') as f:
            content_m =  f.readlines()
        self.img_list = []
        for x in content:
            x =  x.strip()
            self.img_list.append(x)
        self.mask_list = []
        for x in content_m:
            x =  x.strip()
            self.mask_list.append(x)
    def __len__(self):
        return len(self.img_path_list_file)

    def __getitem__(self, index):
        imgfile = self.img_list[index]
        labelfile = self.mask_list[index]
        
        img = Image.open(imgfile).convert('RGB')
        label = np.squeeze(np.load(labelfile))

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, imgfile

        return img, label



class LNENDataSet(data.Dataset):
    def __init__(self, root ='/home/projet16/sources/MCADAForPathonet/segmentation', 
                 img_transform=None, label_transform=None, test=True, split='test',
                 label_type=None, input_ch=3):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        self.img_path_list_file = os.path.join(root, 'iarc_lnen_imgs.txt')
               
        with open(self.img_path_list_file, 'r') as f:
            content =  f.readlines()
        self.img_list = []
        for x in content:
            x =  x.strip()
            self.img_list.append(x)
       
    def __len__(self):
        return len(self.img_path_list_file)

    def __getitem__(self, index):

        imgfile = self.img_list[index]
        
        img = Image.open(imgfile).convert('RGB')
        label = np.zeros((256,256))

        if self.img_transform:
            img = self.img_transform(img)
        #if self.label_transform:
        #    label = self.label_transform(label)

        if self.test:
            return img, label, imgfile
        return img, label



class CityDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True,
                 label_type=None):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir, "leftImg8bit/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "leftImg8bit/%s" % name)
                if label_type == "label16":
                    name = name.replace('leftImg8bit', 'gtFine_label16IDs')
                else:
                    name = name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "gtFine/%s" % name)
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class GTADataSet(data.Dataset):
    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # Note; split "train" and "images" are SAME!!!

        assert split in ["images", "test", "train"]

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root

        imgsets_dir = osp.join(data_dir, "%s.txt" % split)

        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "%s" % name)
                # name = name.replace('leftImg8bit','gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, split="all", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # TODO this does not support "split" parameter

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.test = test

        rgb_dir = osp.join(root, "RGB")
        gt_dir = osp.join(root, "GT", "LABELS16")

        rgb_fn_list = glob.glob(osp.join(rgb_dir, "*.png"))
        gt_fn_list = glob.glob(osp.join(gt_dir, "*.png"))

        for rgb_fn, gt_fn in zip(rgb_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "label": gt_fn
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        img_file = datafiles["rgb"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, label, img_file

        return img, label


class TestDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3):
        assert input_ch == 3
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.listdir(data_dir)
        for name in imgsets_dir:
            img_file = osp.join(data_dir, "%s" % name)
            self.files[split].append({
                "img": img_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return img, 'hoge', img_file
        else:
            return img, img


def get_dataset(dataset_name, split, img_transform, label_transform, test, input_ch=3):
    assert dataset_name in ["gta", "city", "test", "city16", "synthia", "BREAST_TUMOR", "IARC_LNEN"]

    name2obj = {
        "gta": GTADataSet,
        "city": CityDataSet,
        "city16": CityDataSet,
        "synthia": SynthiaDataSet,
        "BREAST_TUMOR":BreastDataSet,
        "IARC_LNEN":LNENDataSet
    }
    ##Note fill in the blank below !! "gta....fill the directory over images folder.
    name2root = {
        "gta": "",  ## Fill the directory over images folder. put train.txt, val.txt in this folder
        "city": "",  ## ex, ./www.cityscapes-dataset.com/file-handling
        "city16": "",  ## Same as city
        "synthia": "",  ## synthia/RAND_CITYSCAPES",
        "BREAST_TUMOR":"",
        "IARC_LNEN":""
    }
    dataset_obj = name2obj[dataset_name]
    root = name2root[dataset_name]

    if dataset_name == "city16":
        return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, label_type="label16")
    if dataset_name == "IARC_LNEN": # TODO check if it's necessary
        return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, label_type="IARC_LNEN")

    return dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                       test=test, input_ch=input_ch)


def check_src_tgt_ok(src_dataset_name, tgt_dataset_name):
    if src_dataset_name == "synthia" and not tgt_dataset_name == "city16":
        raise AssertionError("you must use synthia-city16 pair")
    elif src_dataset_name == "city16" and not tgt_dataset_name == "synthia":
        raise AssertionError("you must use synthia-city16 pair")
    elif  src_dataset_name == "BREAST_TUMOR" and not tgt_dataset_name == "IARC_LNEN":
            raise AssertionError("you must use IARC_LNEN pair")


def get_n_class(src_dataset_name):
    if  src_dataset_name in ["BREAST_TUMOR"]:
        return 4
    if src_dataset_name in ["IARC_LNEN"]:
        return 4
    if src_dataset_name in ["synthia", "city16"]:
        return 16
    elif src_dataset_name in ["gta", "city", "test"]:
        return 20
    else:
        raise NotImplementedError("You have to define the class of %s dataset" % src_dataset_name)
