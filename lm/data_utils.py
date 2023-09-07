import torch
import numpy as np


def get_wikitext2(tokenizer, seed, nsamples=128, seqlen=512, calib=True):
    from datasets import load_dataset
    split = 'validation' if calib else 'test'
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    all_token_masks = tokenizer("\n\n".join(data['text']), return_tensors='np')

    import random
    random.seed(seed)
    np.random.seed(0)

    res = []
    for _ in range(nsamples):
        # pick a random seq : [random_start,random_start+seq_len]
        i = random.randint(0, all_token_masks.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = all_token_masks.input_ids[:, i:j]
        attention_mask = np.ones_like(inp)
        res.append(
            {'input_ids': inp, 'attention_mask': attention_mask})
    return res
