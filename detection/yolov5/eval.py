from tqdm import tqdm
from typing import *
import torch
import json
from detection.yolov5.data_utils import non_max_suppression, rescale_boxes
from detection.coco.utils import encode_results,coco80_to_coco91_class,coco_decetion_eval


def yolov5_coco_eval(test_coco_loader, anno_path, model_forward: Callable, conf_thresh, iou_thresh):
    results = []
    for img_batch, img_ids, cur_hws, ori_hws in tqdm(iterable=test_coco_loader, total=len(test_coco_loader), desc="Evaling"):
        output = model_forward(img_batch)
        boxes = output.cpu().numpy() if isinstance(output, torch.Tensor) else output

        boxes = non_max_suppression(
            boxes, confidence_threshold=conf_thresh, iou_threshold=iou_thresh)
        boxes = rescale_boxes(boxes, cur_hws, ori_hws)

        results.extend(encode_results(boxes_list=boxes,
                                      img_ids=img_ids,
                                      class_map=coco80_to_coco91_class()))

    out_json = './tmp.json'
    with open(file=out_json, mode='w', encoding='utf-8') as file:
        json.dump(results, file)

    coco_decetion_eval(pred_file=out_json, ann_file=anno_path)
