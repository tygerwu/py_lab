import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detection.utils import xyxy2xywh


def coco_decetion_eval(pred_file: str, ann_file: str):
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(pred_file)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def coco80_to_coco91_class():
    # converts 80-index (val2014) to 91-index
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']  # coco80类别


def draw_box(image_path, box_data):
    img = cv2.imread(image_path)
    boxes = box_data[..., :4].astype(np.int32)
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(
            top, left, right, bottom))
        cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(img, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    cv2.imshow('res', img)
    cv2.waitKey(20000)


def encode_results(boxes_list: list[np.ndarray], img_ids: list, class_map):
    res = []
    for i in range(len(img_ids)):
        img_id = img_ids[i]
        boxes = boxes_list[i]
        box_poses = xyxy2xywh(boxes[:, :4])
        # xy center to top-left corner
        box_poses[:, :2] -= box_poses[:, 2:] / 2
        for j in range(box_poses.shape[0]):
            box_pos = box_poses[j]
            box = boxes[j]
            res.append({'image_id': img_id,
                        'category_id': class_map[int(box[5])],
                        'bbox': [float(round(pos, 3)) for pos in box_pos],
                        'score': float(round(box[4], 5))})
    return res
