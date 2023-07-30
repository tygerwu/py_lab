from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
import torchvision.transforms as transforms
from utils import parse_config, seed_all, evaluate
from data.imagenet import load_data
from models import load_model
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy
from my_utils import load_imagenet_from_directory
from ptq import load_calibrate_data
import torch
from torchvision.models import (
    mobilenet_v2, MobileNetV2
)
from ptq import deploy

CALIB_DIR = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/ppq/work/Classification/Data/Imagenet/Calib'
TEST_DIR = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/ppq/work/Classification/Data/Imagenet/Test'
OUT_DIR = './PPQOut'
BATCHSIZE = 32

model = mobilenet_v2(pretrained=True)
# resize to 224x224
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

calib_loader = load_imagenet_from_directory(directory=CALIB_DIR,
                                            transfoms=transform,
                                            batchsize=BATCHSIZE,
                                            shuffle=False,
                                            require_label=False,
                                            num_of_workers=0)


test_loader = load_imagenet_from_directory(directory=TEST_DIR,
                                           transfoms=transform,
                                           batchsize=BATCHSIZE,
                                           shuffle=False,
                                           require_label=True,
                                           num_of_workers=0)

CONFIG_FILE = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/MQBench/application/imagenet_example/PTQ/ptq/mbv2_2_4.yaml'

config = parse_config(CONFIG_FILE)

model = prepare_by_platform(model, BackendType.Academic, config.extra_prepare_dict)

model = model.cuda()


print('begin calibration now!')
cali_data = load_calibrate_data(calib_loader, cali_batchsize=config.quantize.cali_batchsize)

# do activation and weight calibration seperately for quick MSE per-channel for weight one
model.eval()
with torch.no_grad():
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    for batch in cali_data:
        model(batch.cuda())
    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    model(cali_data[0].cuda())

print('begin advanced PTQ now!')
if hasattr(config.quantize, 'reconstruction'):
    model = ptq_reconstruction(model, cali_data, config.quantize.reconstruction)
enable_quantization(model)

evaluate(test_loader, model)
deploy(model, config)
