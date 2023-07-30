# Split Coco Validation Set into Calib and Test set.

import os
import random
import shutil
from tqdm import tqdm


def split(val_dir, calib_out_dir='./img1k_calib', test_out_dir='./img1k_test', calib_size=1, test_size=10):
    for img_class_dir_name in tqdm(os.listdir(val_dir)):
        img_class_dir = os.path.join(val_dir, img_class_dir_name)
        if os.path.isdir(img_class_dir):
            img_files = os.listdir(img_class_dir)

            assert len(img_files) >= calib_size + test_size

            def _create_out_class_dir(out_dir):
                out_class_dir = os.path.join(out_dir, img_class_dir_name)
                if not os.path.exists(out_class_dir):
                    os.makedirs(out_class_dir)
                return out_class_dir

            calib_class_out_dir = _create_out_class_dir(calib_out_dir)
            test_class_out_dir = _create_out_class_dir(test_out_dir)

            # pick imgs randomly
            random_imgs = sorted([(random.random(), file)
                                 for file in img_files])

            calib_set = random_imgs[: calib_size]
            test_set = random_imgs[calib_size: calib_size + test_size]

            for _, file in calib_set:
                source = os.path.join(img_class_dir, file)
                copyto = os.path.join(calib_class_out_dir, file)
                shutil.copyfile(src=source, dst=copyto)

            for _, file in test_set:
                source = os.path.join(img_class_dir, file)
                copyto = os.path.join(test_class_out_dir, file)
                shutil.copyfile(src=source, dst=copyto)


if __name__ == 'main':
    split(val_dir='/media/tyger/Elements/DLData/DataSets/ILSVRC2012/ILSVRC2012_img_val',
          calib_out_dir='/media/tyger/Elements/DLData/DataSets/ILSVRC2012/img1k_calib',
          test_out_dir='/media/tyger/Elements/DLData/DataSets/ILSVRC2012/img1k_test')
