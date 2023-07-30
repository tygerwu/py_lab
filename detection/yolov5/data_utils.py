
import torch
import torchvision
import numpy as np
import cv2

from detection.utils import xywh2xyxy, xyxy2xywh


def resize_and_pad(img: np.ndarray, scale_up=True):
    TARGET_H, TARGET_W = 640, 640
    spatia_shape = img.shape[0:2]
    h, w = spatia_shape[0], spatia_shape[1]

    # scale ratio
    scale_ratio = min(TARGET_H / h, TARGET_W / w)

    # only scale down
    if not scale_up:
        scale_ratio = min(scale_ratio, 1.0)

    scaled_h = int(round(h * scale_ratio))
    scaled_w = int(round(w * scale_ratio))

    # resize
    if (h, w) != (scaled_h, scaled_w):
        img = cv2.resize(img, (scaled_w, scaled_h),
                         interpolation=cv2.INTER_LINEAR)

    # padding
    padding_h = TARGET_H - scaled_h
    padding_w = TARGET_W - scaled_w
    padding_h /= 2
    padding_w /= 2
    # ex : 3 => 1,2
    top, bottom = int(round(padding_h - 0.1)), int(round(padding_h + 0.1))
    left, right = int(round(padding_w - 0.1)), int(round(padding_w + 0.1))

    # why 114 ?
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img


# output shape : [batch_size, box_num, (box_pos, object_confidence, class_probs)]
def non_max_suppression(pred: np.ndarray,
                        confidence_threshold=0.25,
                        iou_threshold=0.45,
                        classes=None,
                        max_detections=300,
                        multi_label=True,
                        agnostic=False
                        ) -> np.ndarray:
    pred = torch.from_numpy(pred)
    max_nms_boxes = 30000
    max_wh = 7680  # maximum box width and height

    batch_size = pred.shape[0]
    output = [torch.zeros((0, 6))] * batch_size

    # filter out boxes with low object confidence
    high_conf_box_ids = pred[..., 4] > confidence_threshold

    for batch_id, boxes_in_batch in enumerate(pred):
        high_conf_boxes = boxes_in_batch[high_conf_box_ids[batch_id]]

        # class-specific confidence score = object confidence * class probablity
        # high_conf_boxes: [box_num, box_attr]
        high_conf_boxes[:, 5:] *= high_conf_boxes[:, 4:5]

        # box_poses: [box_num, 4]
        box_poses = xywh2xyxy(high_conf_boxes[:, :4])

        # decide the class of the box
        # boxes:[box_num,(box_pos, class_conf, class_id)]
        if multi_label:
            # filter out boxes with low class-specific confidence score
            box_ids, class_ids = (high_conf_boxes[:, 5:] > confidence_threshold).nonzero(
                as_tuple=False).T
            boxes = torch.cat(
                (box_poses[box_ids], high_conf_boxes[box_ids, 5 + class_ids, None], class_ids[:, None].float()), dim=1)
        else:
            max_class_confs, class_ids = high_conf_boxes[:, 5:].max(
                1, keepdim=True)
            boxes = torch.cat(
                (box_poses, max_class_confs, class_ids.float()), dim=1)
            # filter
            boxes = boxes[max_class_confs.view(-1) > confidence_threshold]

        if boxes.shape[0] == 0:
            continue

        # filter by class
        if classes is not None:
            boxes = boxes[(boxes[:, 5:6] == np.array(classes)).any(1)]

        # sort by class confidence and remove excess boxes
        boxes = boxes[boxes[:, 4].argsort(descending=True)[:max_nms_boxes]]
        class_offset = boxes[:, 5:6] * (0 if agnostic else max_wh)  # classes
        out = torchvision.ops.nms(
            boxes[:, :4]+class_offset, boxes[:, 4], iou_threshold)
        # limit detections
        out = out[:max_detections]

        output[batch_id] = boxes[out]

    return [out.numpy() for out in output]


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[..., 1])  # x1
        boxes[..., 1].clamp_(0, shape[..., 0])  # y1
        boxes[..., 2].clamp_(0, shape[..., 1])  # x2
        boxes[..., 3].clamp_(0, shape[..., 0])  # y2
    else:  # np.array
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(
            0, shape[..., 1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(
            0, shape[..., 0])  # y1, y2

# Ex: img1_shape[640,640]
#     img0_shape[1080,810]


def scale_boxes(img1_shape, boxes, img0_shape):
    # rescale boxes (xyxy) from img1_shape to img0_shape
    scale_ratio = min(img1_shape[0] / img0_shape[0],
                      img1_shape[1] / img0_shape[1])
    padding = (img1_shape[1] - img0_shape[1] * scale_ratio) / \
        2, (img1_shape[0] - img0_shape[0] * scale_ratio) / 2

    boxes[..., [0, 2]] -= padding[0]  # x padding
    boxes[..., [1, 3]] -= padding[1]  # y padding
    boxes[..., :4] /= scale_ratio
    clip_boxes(boxes, img0_shape)
    return boxes


# boxes : [batch,6]
#   (x0,y0,x1,y1,class_confi,class_id)
# scale_ratios : [batch,1]
# paddings: [batch,4]
#   (top,right)
def scale_boxes(boxes: np.ndarray, scale_ratios: np.ndarray, paddings: np.ndarray):
    boxes[..., [0, 2]] -= paddings[..., [1]]  # x padding
    boxes[..., [1, 3]] -= paddings[..., [0]]  # y padding
    boxes[..., :4] /= scale_ratios
    return boxes


# rescale boxes (xyxy) from cur_hw to ori_hw
# boxes,[batch,[box_num,box_attr]]
#   (x0,y0,x1,y1,class_confi,class_id)
# cur_hw : [batch,2]
# ori_hw:  [batch,2]
def rescale_boxes(boxes: list[np.ndarray], cur_hws: np.ndarray, ori_hws: np.ndarray):
    for i in range(len(boxes)):
        box = boxes[i]
        cur_hw = cur_hws[i]
        ori_hw = ori_hws[i]
        scale_ratio = min(cur_hw[0] / ori_hw[0], cur_hw[1] / ori_hw[1])
        padding_h = (cur_hw[0] - ori_hw[0] * scale_ratio) / 2
        padding_w = (cur_hw[1] - ori_hw[1] * scale_ratio) / 2

        box[..., [0, 2]] -= padding_w  # x padding
        box[..., [1, 3]] -= padding_h  # y padding
        box[..., : 4] /= scale_ratio
        clip_boxes(box, ori_hw)
    return boxes


def load_and_preprocess(img_path):
    img = cv2.imread(img_path)
    ori_hw = img.shape[0: 2]
    img = resize_and_pad(img.copy())
    cur_hw = img.shape[0: 2]
    img = img.astype(np.float32)
    # HWC -> CHW
    img = img.transpose((2, 0, 1))[:: -1]
    # CHW -> NCHW
    img = np.expand_dims(img, axis=0)

    # Normalize
    img /= 255.0
    img = np.ascontiguousarray(img)

    def _to_batch(list_data):
        return np.expand_dims(np.array(list_data), axis=0)

    return img, _to_batch(cur_hw), _to_batch(ori_hw)
