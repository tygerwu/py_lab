import os
import numpy as np
import onnxruntime
from detection.yolov5.data_utils import load_and_preprocess, non_max_suppression, rescale_boxes
from detection.coco.utils import draw_box
from detection.yolov5.eval import yolov5_coco_eval
from dataset import coco_dataloader
import json


class OnnxYolov5():
    def __init__(self, onnxpath, use_cuda=False):
        providers = ['TensorrtExecutionProvider'] if use_cuda else [
            'CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(
            onnxpath, providers=providers)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
        return input_feed

    def demo(self, img_path, draw_box=True, conf_thresh=0.25, iou_thresh=0.45):
        img, cur_hw, ori_hw = load_and_preprocess(img_path)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        boxes = non_max_suppression(
            pred, iou_threshold=iou_thresh, confidence_threshold=conf_thresh)
        boxes = rescale_boxes(
            boxes=boxes, cur_hws=cur_hw, ori_hws=ori_hw)
        if draw_box:
            draw_box(img_path, boxes[0])
        return boxes

    def batch_infer(self, img_paths: list[str]):
        imgs = []
        cur_hws = []
        ori_hws = []
        for img_path in img_paths:
            img, cur_hw, ori_hw = load_and_preprocess(img_path)
            imgs.append(img)
            cur_hws.append(cur_hw)
            ori_hws.append(ori_hw)
        imgs = np.concatenate(imgs, axis=0)
        ori_hws = np.concatenate(ori_hws, axis=0)
        cur_hws = np.concatenate(cur_hws, axis=0)
        input_feed = self.get_input_feed(imgs)
        batch_pred = self.onnx_session.run(None, input_feed)[0]
        boxes = non_max_suppression(
            batch_pred)
        boxes = rescale_boxes(
            boxes=boxes, cur_hws=cur_hws, ori_hws=ori_hws)
        print(boxes)
        return boxes

    def coco_eval(self, test_img_dir, anno_path, conf_thresh, iou_thresh):
        coco_loader = coco_dataloader(
            img_dir=test_img_dir,
            only_img=False,
            batch_size=32
        )

        def _model_forward(input):
            input_feed = self.get_input_feed(input)
            pred = self.onnx_session.run(None, input_feed)[0]
            return pred
        yolov5_coco_eval(test_coco_loader=coco_loader,
                         anno_path=anno_path, model_forward=_model_forward, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        return


if __name__ == "__main__":
    resource_dir = '/media/tyger/linux_ssd/codes/python_test/Torch/object_dect/yolov5/resource/'
    onnx_path = resource_dir + 'yolov5s.onnx'
    img0_path = resource_dir + 'bus.jpg'
    img1_path = resource_dir + 'demo1.jpg'
    model = OnnxYolov5(onnx_path, use_cuda=False)
    # model.demo(
    #     '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/datasets/coco/images/val2017/000000005503.jpg')
    # model.batch_infer([img0_path, img1_path])
    coco_dir = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/datasets/coco'
    ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    val_dir = os.path.join(coco_dir, 'images', 'val2017')
    conf_thresh = 0.001
    iou_thresh = 0.65
    model.coco_eval(val_dir, ann_file, conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh)
