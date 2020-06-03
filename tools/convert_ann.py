import sys
import logging
import os
import numpy as np
from threading import Thread
import time
from path import Path
import json
import click
import sys
from utils import get_matrix, get_pose
from convert_for_det import get_det_cls_targets, get_regression_targets, get_detection_regression_targets, \
    get_identity_masks, get_tc_matrix
import multiprocessing as mp

assert sys.version_info >= (3, 6), '[!] This script requires Python >= 3.6'

width = 32
height = 32
stride = 16
shift_x = np.arange(0, width * stride, stride)
shift_y = np.arange(0, height * stride, stride)
shift_y, shift_x = np.meshgrid(shift_y, shift_x)
shift_x = np.reshape(shift_x, -1)
shift_y = np.reshape(shift_y, -1)
points = np.stack([shift_x + stride // 2, shift_y + stride // 2], axis=1)


def create_data(ann: str, out_file_targets: str, out_file_targets_cls: str, out_file_targets_reg: str,
                out_file_targets_tc: str, clip_length: int, video_length: int):
    idxs = video_length - clip_length + 1
    video_file = os.path.splitext(os.path.basename(ann))
    out_file = '{}.log'.format(video_file[0])
    with open(ann, 'r') as json_file:
        data = json.load(json_file)
    data = np.array(data)

    # idxs_dict = dict()
    total = []
    final_targets = np.empty((idxs), dtype=object)
    final_targets_cls = np.empty((idxs), dtype=object)
    final_targets_reg = np.empty((idxs), dtype=object)
    final_targets_tc = np.empty((idxs), dtype=object)
    for idx in range(0, idxs):
        frames_dict = dict()
        total_tracks = list()
        for i in range(0, clip_length):
            frame_data = data[data[:, 0] == i + idx + 1]
            frames_dict[i + idx] = dict()
            for x, person_id in enumerate(frame_data[:, 1], 0):
                pose = get_pose(frame_data=frame_data, person_id=person_id)
                if pose.invisible:
                    continue
                x_min, y_min, w, h = pose.bbox_2d
                frames_dict[i + idx][person_id] = [x_min, y_min, x_min + w, y_min + h]
                total_tracks.append(person_id) if person_id not in total_tracks else total_tracks

        try:
            targets = get_matrix(frames_dict, total_tracks)
            targets = np.expand_dims(targets, 0)
            bbox_identities = get_identity_masks(targets, points, height, width)
            cls_targets = get_det_cls_targets(bbox_identities)
            reg_targets = get_detection_regression_targets(bbox_identities, targets)
            tc_targets = get_tc_matrix(reg_targets, clip_length)
            final_targets[idx] = targets
            final_targets_cls[idx] = cls_targets
            final_targets_reg[idx] = reg_targets
            final_targets_tc[idx] = tc_targets
        except:
            logname = '{}.log'.format(os.path.basename(out_file))
            logging.basicConfig(filename=logname,
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
            logging.debug('Video file is {}.'.format(video_file))
            logging.debug('targets Size: {}'.format(','.join([str(x) for x in list(targets.shape)])))
            logging.debug('points Size: {}'.format(','.join([str(x) for x in list(points.shape)])))
        # total.append([targets, cls_targets, reg_targets])
        '''
        print(targets.shape)
        print(cls_targets.shape)
        print(reg_targets.shape)
        # idxs_dict[idx] = labels
        '''
    '''
    final_targets = np.empty(len(total))
    final_targets_cls = np.empty(len(total))
    final_targets_reg = np.empty(len(total))
    for idx, arr in enumerate(total):
        final_targets[idx] = total[0][0]
        final_targets_cls[idx] = total[0][1]
        final_targets_reg[idx] = total[0][2]
    '''
    np.savez(out_file_targets, final_targets)
    np.savez(out_file_targets_cls, final_targets_cls)
    np.savez(out_file_targets_reg, final_targets_reg)
    np.savez(out_file_targets_tc, final_targets_tc)
    '''
    np.savez(out_file_targets, total[0])
    np.savez(out_file_targets_cls, total[1])
    np.savez(out_file_targets_reg, total[2])
    '''
    return None


class FrameDataCreatorThread(Thread):
    def __init__(self, in_path: str, out_path: str, clip_length: int, video_length: int):
        Thread.__init__(self)
        self.in_path = in_path
        self.out_path = out_path
        self.clip_length = clip_length
        self.video_length = video_length

    def run(self):
        print('[{}] > START'.format(self.out_path))
        create_data(self.in_path, self.out_path, self.clip_length, self.video_length)
        print('[{}] > DONE'.format(self.out_path))


H1 = 'directory where you want to save the numpy annotations'


@click.command()
@click.option('--out_dir_path', type=str, default='new_ann_8', help=H1)
@click.option('--clip_length', type=int, default=8)
@click.option('--video_length', type=int, default=900)
def main(out_dir_path, clip_length=8, video_length=900):
    annotations = list()
    out_subdir_path_targets = list()
    out_subdir_path_targets_cls = list()
    out_subdir_path_targets_reg = list()
    out_subdir_path_targets_tc = list()

    for s in Path('annotations').dirs():
        annotations_temp = s.files()
        temp = os.path.join(out_dir_path, s.basename().split('.')[0], 'targets')
        os.makedirs(temp, exist_ok=True)
        temp = os.path.join(out_dir_path, s.basename().split('.')[0], 'cls_targets')
        os.makedirs(temp, exist_ok=True)
        temp = os.path.join(out_dir_path, s.basename().split('.')[0], 'reg_targets')
        os.makedirs(temp, exist_ok=True)
        temp = os.path.join(out_dir_path, s.basename().split('.')[0], 'tc_targets')
        os.makedirs(temp, exist_ok=True)
        annotations += annotations_temp
        out_subdir_path_targets += [os.path.join(out_dir_path, s.basename(), 'targets', x.basename().split('.')[0]) for
                                    x in
                                    annotations_temp]
        out_subdir_path_targets_cls += [
            os.path.join(out_dir_path, s.basename(), 'cls_targets', x.basename().split('.')[0]) for x in
            annotations_temp]
        out_subdir_path_targets_reg += [
            os.path.join(out_dir_path, s.basename(), 'reg_targets', x.basename().split('.')[0]) for x in
            annotations_temp]
        out_subdir_path_targets_tc += [
            os.path.join(out_dir_path, s.basename(), 'hng_targets', x.basename().split('.')[0]) for x in
            annotations_temp]

    CLIP_LENGTH = [clip_length] * len(annotations)
    VIDEO_LENGTH = [video_length] * len(annotations)
    arguments = list(zip(annotations, out_subdir_path_targets, out_subdir_path_targets_cls, out_subdir_path_targets_reg,
                         out_subdir_path_targets_tc, CLIP_LENGTH, VIDEO_LENGTH))
    p = mp.Pool()
    p.starmap(create_data, arguments)


if __name__ == "__main__":
    main()
