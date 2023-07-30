
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from classification.imagenet.dataset import load_imagenet_from_directory
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass,
                                    ParameterQuantizePass,
                                    RuntimeCalibrationPass)
from quant.ppq_quantizer import Int8Quantizer
from ppq.executor import TorchExecutor
from ppq.core import TargetPlatform
from ppq.api import (
    ENABLE_CUDA_KERNEL,
    DISABLE_CUDA_KERNEL,
    dump_torch_to_onnx,
    load_onnx_graph
)
from ppq import (
    layerwise_error_analyse
)
from classification.imagenet.eval import imagenet_eval
import ppq.lib as PFL
import os
import torch
import torchvision.transforms as transforms


CALIB_DIR = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/ppq/work/Classification/Data/Imagenet/Calib'
TEST_DIR = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/ppq/work/Classification/Data/Imagenet/Test'
OUT_DIR = '/media/tyger/linux_ssd/codes/githubs/model_compression/quant/ppq/work/Classification/PPQOut'
BATCHSIZE = 32
DEVICE = 'cuda'
# tuneable parameters
BITS = 8
PER_CHANNEL = False
SYMMETRICAL = False
QUANT_TYPES = ('Conv', 'MatMul', 'Gemm', 'PPQBiasFusedMatMul')
NOT_QUANT_LAYERS = ('/features/features.1/conv/conv.0/conv.0.0/Conv',
                    '/features/features.15/conv/conv.1/conv.1.0/Conv')

# resize to 224x224
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights)
model.eval()
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


with ENABLE_CUDA_KERNEL():
    # convert to onnx model
    fp_onnx_model_name = 'mobilenetv2_fp32.onnx'
    fp_onnx_model_path = os.path.join(OUT_DIR, fp_onnx_model_name)
    input_shape = [BATCHSIZE, 3, 224, 224]

    dump_torch_to_onnx(model=model,
                       onnx_export_file=fp_onnx_model_path,
                       input_shape=input_shape,
                       device='cpu')

    graph = load_onnx_graph(onnx_import_file=fp_onnx_model_path)

    # create quantizer
    quantizer = Int8Quantizer(graph=graph,
                              sym=SYMMETRICAL,
                              num_of_bits=BITS,
                              per_channel=PER_CHANNEL)

    # convert op to quantable-op
    for name, op in graph.operations.items():
        if op.type in QUANT_TYPES and name not in NOT_QUANT_LAYERS:
            quantizer.quantize_operation(name, platform=TargetPlatform.INT8)

    # build quant pipeline.
    pipeline = PFL.Pipeline([
        # LayerwiseEqualizationPass(iterations=10),
        ParameterQuantizePass(),
        RuntimeCalibrationPass(),
        # LearnedStepSizePass(steps=500, collecting_device=DEVICE, block_size=5)
    ])

    # call pipeline.
    executor = TorchExecutor(graph=graph, device=DEVICE)
    executor.tracing_operation_meta(
        torch.zeros(size=[BATCHSIZE, 3, 224, 224]))

    pipeline.optimize(graph=graph,
                      dataloader=calib_loader,
                      verbose=True,
                      calib_steps=32,
                      collate_fn=lambda x: x.to(DEVICE),
                      executor=executor)

    quant_executor = TorchExecutor(graph=quantizer._graph, device=DEVICE)
    # evaluation

    def model_forward_function(input_tensor): return torch.tensor(
        quant_executor(*[input_tensor.to(DEVICE)])[0])

    imagenet_eval(test_img_loader=test_loader,
                  model_forward=model_forward_function)

    # layerwise_error_analyse(graph=quantizer._graph,
    #                         running_device=DEVICE,
    #                         collate_fn=lambda x: x[0].to(DEVICE),
    #                         dataloader=test_loader)
