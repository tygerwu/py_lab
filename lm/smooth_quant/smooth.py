import torch
from typing import Callable
import functools
import torch.nn as nn
from tqdm import tqdm


def get_act_max(model, sample_num, get_tokens: Callable):
    act_scales = {}
    device = next(model.parameters()).device

    def _forward_hook(module, inputs, outputs, name):
        activation = inputs
        if isinstance(inputs, tuple):
            activation = inputs[0]
        # [batch,seq_len,ic]
        ic = activation.shape[-1]
        # [batchxseq_len,ic]
        activation = activation.view(-1, ic).abs().detach()
        # Get max value along ic.
        # [ic]
        comming_max = torch.max(activation, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(_forward_hook, name=name))
            )

    for i in tqdm(range(sample_num)):
        tokens = get_tokens(i)
        model(tokens.to(device))

    for h in hooks:
        h.remove()

    return act_scales


def get_weight_max(liner_layer: nn.Linear):
    # Weight : [oc,ic]
    return liner_layer.weight.abs().max(dim=0)[0]


def compute_scale(weight_max, act_max, alpha):
    return (act_scales.pow(alpha) / weight_max.pow(1-alpha)).clamp(min=1e-5)


if __name__ == '__main__':
    from lm.data_utils import get_wikitext2
    from transformers import AutoTokenizer, AutoModel

    hf_model = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModel.from_pretrained(hf_model)
    calib_num = 128
    seq_len = 512
    calib_data = get_wikitext2(tokenizer, 20, nsamples=calib_num, seqlen=512)

    def _get_calib_tokens(i):
        return torch.from_numpy(calib_data[i]['input_ids'])

    act_scales = get_act_max(model, calib_num, _get_calib_tokens)
