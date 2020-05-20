import os

import torch
from torchvision import datasets
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
from skimage import io
from boxlist import BoxList


def has_only_empty_bbox(annot):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True


def convert_to_xyxy(target):
    xmins = target[:, :, 0]
    ymins = target[:, :, 1]
    w = target[:, :, 2]
    h = target[:, :, 3]
    xmaxs = xmins + w
    ymaxs = ymins + h
    xyxy = target
    xyxy[:, :, 2] = xmaxs
    xyxy[:, :, 3] = ymaxs
    return xyxy


class JTADataset(Dataset):
    """
    JTA Dataset handler
    """

    def __init__(self, path, split, clip_length=8, transform=None):
        self.root = path
        self.videos_path = os.path.join(self.root, 'frames', split)
        self.anns_path = os.path.join(self.root, 'anns_%s' % clip_length, split)
        self.transform = transform
        self.clip_length = clip_length
        self.df = 900
        self.videos = self.get_videos()
        self.anns = self.get_anns()
        self.length = self.get_length()

    def get_videos(self):
        videos_folders = sorted(glob.glob(
            os.path.join(self.videos_path, 'seq_*'),
            recursive=False
        ))
        return videos_folders

    def get_anns(self):
        anns = sorted(glob.glob(
            os.path.join(self.anns_path, '*.npz'),
            recursive=False
        ))
        return anns

    def get_length(self):
        num_videos = len(self.videos)
        length = (self.df - self.clip_length + 1) * num_videos
        return length

    def get_data(self, index):
        video_index, starting_frame = divmod(index, self.df - self.clip_length + 1)
        starting_frame += 1
        video_folder = self.videos[video_index]
        frames = [
            os.path.join(video_folder, '{}.jpg'.format(x))
            for x in range(starting_frame, starting_frame + self.clip_length)
        ]
        labels = self.get_annotation_frames(video_index, starting_frame)

        return frames, labels

    def get_annotation_frames(self, video_index, starting_frame):
        annotation_file = self.anns[video_index]
        data = np.load(annotation_file)
        labels = data['arr_{}'.format(starting_frame - 1)]

        return labels

    def __getitem__(self, idx):
        """
        Generates one sample of data
        :param idx:
        :return: dictionary of clips and labels
            clips: list of images
            labels: mat of shape Frames x Tracklets x Bbs
        """
        frames, target = self.get_data(idx)
        clip = [io.imread(frame) for frame in frames]
        if self.transform:
            clip, target = self.transform(clip, target)

        return clip, target, idx

    def __len__(self):
        return self.length


class COCODataset(datasets.CocoDetection):
    def __init__(self, path, split, transform=None):
        root = os.path.join(path, f'{split}2017')
        annot = os.path.join(path, 'annotations', f'instances_{split}2017.json')

        super().__init__(root, annot)

        self.ids = sorted(self.ids)

        if split == 'train':
            ids = []

            for id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
                annot = self.coco.loadAnns(ann_ids)

                if has_valid_annotation(annot):
                    ids.append(id)

            self.ids = ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.transform = transform

    def __getitem__(self, index):
        img, annot = super().__getitem__(index)

        annot = [o for o in annot if o['iscrowd'] == 0]

        boxes = [o['bbox'] for o in annot]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [o['category_id'] for o in annot]
        classes = [self.category2id[c] for c in classes]
        classes = torch.tensor(classes)
        target.fields['labels'] = classes

        target.clip(remove_empty=True)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, index

    def get_image_meta(self, index):
        id = self.id2img[index]
        img_data = self.coco.imgs[id]

        return img_data


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)


def collate_fn(config):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], config.size_divisible)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data
