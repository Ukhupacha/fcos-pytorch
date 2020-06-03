import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from skimage.draw import circle, rectangle, rectangle_perimeter
from skimage import img_as_ubyte
from pose import Pose
from joint import Joint
import time
import torch
import colorsys
from numba import jit


def get_pose(frame_data, person_id):
    # type: (np.ndarray, int) -> Pose
    """
    :param frame_data: data of the current frame
    :param person_id: person identifier
    :return: list of joints in the current frame with the required person ID
    """
    pose = [Joint(j) for j in frame_data[frame_data[:, 1] == person_id]]
    pose.sort(key=(lambda j: j.type))
    return Pose(pose)


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    plt.ion()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


def draw_bb(images, labels):
    """
    Draws bounding boxes on the given images
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    labels: List of labels for each frame.
    """

    labels = labels[0]

    for image, label in zip(images, labels):
        N = len(label)
        HSV_tuples = [(x * 255.0 / N, 100, 100) for x in range(N)]
        RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
        for i, bbs in enumerate(label):
            for n in range(0, 10):
                delta = torch.tensor([n])
                start = (bbs[1] + delta, bbs[0] + delta)
                end = (bbs[1] + bbs[3] - delta, bbs[0] + bbs[2] - delta)
                rr, cc = rectangle_perimeter(start, end=end, clip=True, shape=image.shape)
                image[rr, cc] = RGB_tuples[i]


def show_images_batch(sample_batched):
    """Show image with labels for a batch of samples."""
    images_batch, labels = sample_batched['clip'], sample_batched['labels']
    im_list = [i.numpy().transpose((1, 2, 0)) for i in images_batch[0]]
    draw_bb(im_list, labels)
    fig = plt.figure()
    plt.ion()
    images = []
    for i in range(len(im_list)):
        img_plot = plt.imshow(im_list[i])
        images.append([img_plot])
    ani = animation.ArtistAnimation(fig, images, blit=True, repeat=False)
    plt.show(block=False)
    plt.pause(3)
    plt.close('all')


def get_matrix(annotation_dict, tracks_dict):
    """
    Returns matrix of labels given the annotation dict and tracks dict
    :return: Matrix of shape Frames x Tracklets x BBs
    """
    frames_ids = list(annotation_dict.keys())
    tracks_ids = list(tracks_dict)
    mat = np.full((len(frames_ids), len(tracks_ids), 4), -1)

    for i, frame_id in enumerate(frames_ids, 0):
        for x, track_id in enumerate(tracks_ids, 0):
            if track_id in annotation_dict[frame_id]:
                x_min, y_min, w, h = annotation_dict[frame_id][track_id]
                # Update x_min
                mat[i, x, 0] = x_min
                # Update y_min
                mat[i, x, 1] = y_min
                # Update w
                mat[i, x, 2] = w
                # Update h
                mat[i, x, 3] = h
    return mat
