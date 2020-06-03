import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def point_inside_box(targets, locations):
    targets = np.squeeze(targets, 0)
    targets = np.expand_dims(targets, 1)
    locations = np.expand_dims(locations, 0)
    locations = np.expand_dims(locations, 2)
    locations = np.repeat(locations, targets.shape[2], 2)
    c1 = locations[:, :, :, 0] <= targets[:, :, :, 2]
    c2 = locations[:, :, :, 0] >= targets[:, :, :, 0]
    c3 = locations[:, :, :, 1] <= targets[:, :, :, 3]
    c4 = locations[:, :, :, 1] >= targets[:, :, :, 1]
    mask = c1 & c2 & c3 & c4
    return mask


def visualize_mask(mask, height, width):
    num_frames = mask.shape[2]
    num_tracklets = mask.shape[-1]
    mask = np.reshape(mask, (num_frames, height, width, num_tracklets))
    mask = 255 * mask
    mask = mask.astype(np.uint8)
    counter = 1
    for rownum, framenum in enumerate(range(num_frames)):
        for colnum, tracklet_num in enumerate(range(num_tracklets)):
            vis_data = np.zeros((height, width), dtype=np.uint8)
            vis_data += mask[framenum, :, :, tracklet_num]
            ax = plt.subplot('{}{}{}'.format(num_frames, num_tracklets, counter))
            ax.imshow(vis_data)
            counter += 1

    plt.show()
    plt.close()
    return None


def get_boxes_area(targets):
    xmins = targets[:, :, :, 0]
    ymins = targets[:, :, :, 1]
    xmaxs = targets[:, :, :, 2]
    ymaxs = targets[:, :, :, 3]
    areas = (ymaxs - ymins + 1) * (xmaxs - xmins + 1)
    return areas


def is_point_inside(location, bbox):
    point_x, point_y = location
    c1 = point_x <= bbox[2]
    c2 = point_x >= bbox[0]
    c3 = point_y <= bbox[3]
    c4 = point_y >= bbox[1]
    decision = c1 & c2 & c3 & c4
    decision = 1 * decision
    decision = decision.astype(np.float32)
    return decision


def get_identity_masks(targets, locations, height, width):
    initial_mask = point_inside_box(targets, locations)
    initial_mask = np.reshape(initial_mask, (initial_mask.shape[0], height, width, -1))

    frame_num, i, j, tracklet_num = np.where(initial_mask)
    inew = i * 16 + 8
    jnew = j * 16 + 8
    valid_locations = np.stack([frame_num, i, j, tracklet_num, inew, jnew], axis=0)
    output = get_bbox_identities(valid_locations, targets, targets.shape[1], height, width)
    return output


def get_bbox_identities(valid_locations, targets, num_frames, height, width):
    output = np.zeros((1, 1, num_frames, height, width))
    boxes_area = get_boxes_area(targets)
    for i in range(valid_locations.shape[1]):
        valid_boxes = targets[0, valid_locations[0, i], :, :].tolist()
        boxes_mask = list(
            map(
                lambda x: is_point_inside(valid_locations[4:, i], x),
                valid_boxes
            )
        )
        boxes_mask = [x * y for x, y in zip(boxes_mask, boxes_area[0, valid_locations[0, i], :].tolist())]
        nonzero_ind = [i for i, _ in enumerate(boxes_mask)]
        valid_boxes_mask = [i for i in boxes_mask if i > 0.0]
        if len(valid_boxes_mask) == 0:
            continue
        min_area = min(valid_boxes_mask)
        valid_boxes_ind = [i for i, e in enumerate(boxes_mask) if e == min_area][0]
        output[0, 0, valid_locations[0, i], valid_locations[1, i], valid_locations[2, i]] = valid_boxes_ind + 1

    return output


def get_det_cls_targets(bbox_identities):
    output = bbox_identities.astype(np.bool)
    return output


def get_regression_targets(bbox_identities, targets):
    output = np.zeros((1, 4, targets.shape[1], bbox_identities.shape[-2], bbox_identities.shape[-1]))
    num_tracklets = targets.shape[-2]
    u, indices = np.unique(bbox_identities, return_inverse=True)
    tracklets_common = u[np.argmax(
        np.apply_along_axis(np.bincount, 2, indices.reshape(bbox_identities.shape), None, np.max(indices) + 1), axis=2)]
    tracklets_common = np.reshape(tracklets_common, -1).tolist()
    for ind, val in enumerate(tracklets_common):
        if val == 0.0:
            continue
        row, col = np.unravel_index(ind, (bbox_identities.shape[-2], bbox_identities.shape[-1]))
        for i, val_store in enumerate(bbox_identities[0, 0, :, row, col].tolist()):
            if val_store == val:
                # print(i)
                # print(val)
                output[0, :, i, row, col] = targets[0, i, int(val) - 1, :]
    return output


def get_detection_regression_targets(bbox_identities, targets):
    # print(bbox_identities.shape)
    # print(targets.shape)
    output = np.zeros((1, 4, targets.shape[1], bbox_identities.shape[-2], bbox_identities.shape[-1]))
    num_tracklets = targets.shape[-2]
    for i in range(num_tracklets):
        locations = np.argwhere(bbox_identities == i + 1)
        # print(locations)
        output[0, :, locations[:, 2], locations[:, -2], locations[:, -2]] = targets[0, locations[:, 2], i, :]

    return output


def get_tc_matrix(reg_targets, clip_length):

    print(reg_targets)
    print(reg_targets.shape)
    print(clip_length)


    #num_tracks = clip.shape[2]
    #mat = np.identity(num_tracks)
    return None

# if __name__ == "__main__":
#     '''
#     THE FOLLOWING PART IS USED TO CREATE THE LOCATIONS FOR GENERATING THE TARGET ASSIGNMENT
#     '''
#     width = 32
#     height = 32
#     stride = 16
#     shift_x = np.arange(0, width * stride, stride)
#     shift_y = np.arange(0, height * stride, stride)
#     shift_y, shift_x = np.meshgrid(shift_y, shift_x)
#     shift_x = np.reshape(shift_x, -1)
#     shift_y = np.reshape(shift_y, -1)
#     points = np.stack([shift_x + stride // 2, shift_y + stride // 2], axis=1)
#     bboxes = np.array([[[30, 40, 50, 150], [45, 35, 60, 155]], [[40, 40, 60, 170], [45, 35, 60, 155]]])
#     targets = np.expand_dims(bboxes, 0)
#     initial_mask = point_inside_box(targets, points)
#     #visualize_mask(initial_mask, height, width)
#     #plt.show()
#     #plt.close()
#     bbox_identities = get_identity_masks(targets, points, height, width)
#     '''
#     WHAT YOU NEED FOR DETECTION TARGET ASSIGNMENT
#
#     '''
#     cls_targets = get_det_cls_targets(bbox_identities)
#     reg_targets = get_detection_regression_targets(bbox_identities, targets)
#     '''
#     num_frames = output.shape[2]
#     for framenum in range(num_frames):
#         data = output[0, 0, framenum, :, :]
#         ax = plt.subplot('1{}{}'.format(num_frames, framenum + 1))
#         ax.imshow(data)
#
#     plt.show()
#     plt.close()
#     '''
