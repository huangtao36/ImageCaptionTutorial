# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 16:49
# @Author  : Tao
# @Project : ImageCaption
# @File    : data_loader.py
"""
Use for load COCO dataset for Image Caption task
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import nltk
from PIL import Image
import json


def anno_loader(file_):
    """
    use to load coco caption annotations
    :param file_: annotation file abspath
    :return: anns, a dict,
        the key is the caption id and the value is annotation,
        each annotation also a dict with three keys: 'caption', 'id', 'image_id'
    """
    dataset = json.load(open(file_, 'r'))
    anns = {}
    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            anns[ann['id']] = ann

    imgs = {}
    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img
    return anns, imgs


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_file, vocab_, transform_=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json_file: coco annotation file path.
            vocab_: vocabulary wrapper.
            transform_: image transformer.
        """
        self.root = root
        self.anns, self.imgs = anno_loader(file_=json_file)
        self.ids = list(self.anns.keys())
        self.vocab = vocab_
        self.transform = transform_

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        anns = self.anns

        # noinspection PyShadowingNames
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = anns[ann_id]['caption']
        img_id = anns[ann_id]['image_id']
        img_name = self.imgs[img_id]['file_name']

        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data_):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    collate_fn这个函数的输入是一个list，
    list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    总体来讲就是先按label长度进行排序，然后进行长度的pad，
    最后输出图片，label以及每个label的长度的list

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    :param data_: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    :return: images: torch tensor of shape (batch_size, 3, 256, 256).
             targets: torch tensor of shape (batch_size, padded_length).
             lengths: list; valid length for each padded caption.
    """

    # Sort a data list by caption length (descending order).
    data_.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data_)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json_file, vocab_, transform_,
               batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json_file=json_file,
                       vocab_=vocab_,
                       transform_=transform_)

    data_ = torch.utils.data.DataLoader(
        dataset=coco,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_


if __name__ == '__main__':
    """
    仅用于测试本程序文件, Load的数据可直接用于训练，参考使用
    默认此文件位于工程根目录下
    """

    """文件路径"""
    project_per_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    project_path = os.path.abspath(os.path.join(os.getcwd(), '../'))
    dataset_root_path = os.path.join(
        project_per_path, 'Dataset\ImageCaption\COCO')

    vocab_path = os.path.join(project_path, 'data/vocal.pkl')
    image_dir = os.path.join(dataset_root_path, 'train2014')
    caption_file = os.path.join(project_path, 'data/captions_train2014.json')

    """transfrom"""
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    """
    Load vocabulary
    如果要Load vocabulary, 必须导入 Vocabulary 类
    """
    from utils.util import Vocabulary  # It must load this Class
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print("Vocab size: ", len(vocab))

    """Build data loader"""
    data_loader = get_loader(image_dir, caption_file, vocab, transform,
                             batch_size=128, shuffle=True, num_workers=2)

    """Loop data"""
    for i, (images, captions, lengths) in enumerate(data_loader):
        print("images.shape: ", images.shape)
        print("captions.shape: ", captions.shape)
        print("lengths: ", lengths)

        break
