import torch
from typing import Callable
import functools
import torch.nn as nn
from tqdm import tqdm


def get_act_scales(model, sample_num, get_tokens: Callable):
    act_scales = {}
    device = next(model.parameters()).device

    def _forward_hook(module, inputs, outputs, name):
        activation = inputs
        if isinstance(inputs, tuple):
            activation = inputs[0]
        # [Batch,SeqLen,IC]
        ic = activation.shape[-1]
        # [BatchxSeqLen,IC]
        activation = activation.view(-1, ic).abs().detach()
        # Get max value along IC.
        # [IC]
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

    act_scales = get_act_scales(model, calib_num, _get_calib_tokens)
