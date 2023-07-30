# Split Coco Validation Set into Calib and Test set.
import json
import os
import random
import shutil


def split(val_img_dir, annotation_path,
          calib_out_dir='./mscoco_calib',
          test_out_dir='./mscoco_test',
          calib_size=512,
          test_size=4096):
    img_names = [img for img in os.listdir(val_img_dir)]

    assert len(img_names) >= calib_size + test_size

    # pick imgs randomly
    random_imgs = sorted([(random.random(), img_name)
                          for img_name in img_names])

    def _create_out_dir(out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    _create_out_dir(calib_out_dir)
    _create_out_dir(test_out_dir)

    calib_set = random_imgs[: calib_size]
    test_set = random_imgs[calib_size: calib_size + test_size]

    calib_img_names = []
    for _, img_name in calib_set:
        calib_img_names.append(img_name)
        source = os.path.join(val_img_dir, img_name)
        copyto = os.path.join(calib_out_dir, img_name)
        shutil.copyfile(src=source, dst=copyto)

    test_img_names = []
    for _, img_name in test_set:
        test_img_names.append(img_name)
        source = os.path.join(val_img_dir, img_name)
        copyto = os.path.join(test_out_dir, img_name)
        shutil.copyfile(src=source, dst=copyto)

    # split annotations
    with open(annotation_path, 'r', encoding='utf-8') as file:
        j_obj = json.load(file)
        annotations = j_obj['annotations']
        images = j_obj['images']

        calib_img, test_img, calib_ann, test_ann = [], [], [], []
        id_to_imgname = {}
        for img in images:
            file_name = img['file_name']
            id_to_imgname[img['id']] = file_name

            if file_name in calib_img_names:
                calib_img.append(img)

            if file_name in test_img_names:
                test_img.append(img)

        for ann in annotations:
            if id_to_imgname[ann['image_id']] in calib_img_names:
                calib_ann.append(ann)

            if id_to_imgname[ann['image_id']] in test_img_names:
                test_ann.append(ann)

    # store annotation file to file.
    with open(os.path.join(calib_out_dir, 'mscoco_calib_annotations.json'), 'w', encoding='utf-8') as file:
        j_obj['annotations'] = calib_ann
        j_obj['images'] = calib_img
        json.dump(j_obj, file)

    with open(os.path.join(test_out_dir, 'mscoco_test_annotations.json'), 'w', encoding='utf-8') as file:
        j_obj['annotations'] = test_ann
        j_obj['images'] = test_img
        json.dump(j_obj, file)


if __name__ == '__main__':
    val_img_dir = '/media/tyger/Elements/DLData/DataSets/coco2017/val2017'
    annotation_path = '/media/tyger/Elements/DLData/DataSets/coco2017/annotations_trainval2017/annotations/instances_val2017.json'
    calib_out_dir = '/media/tyger/Elements/DLData/DataSets/coco2017/ptq/mscoco_calib'
    test_out_dir = '/media/tyger/Elements/DLData/DataSets/coco2017/ptq/mscoco_test'
    split(val_img_dir, annotation_path, calib_out_dir, test_out_dir)
