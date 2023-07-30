from tqdm import tqdm
import torch.utils.data
import torch
from typing import Callable


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def imagenet_eval(test_img_loader, model_forward: Callable):
    recorder = {
        'loss': [],
        'top1_accuracy': [],
        'top5_accuracy': [],
    }
    loss_fn = torch.nn.CrossEntropyLoss().to('cpu')

    for batch_input, batch_label in tqdm(test_img_loader, desc='Evaluating Model...', total=len(test_img_loader)):

        batch_input = batch_input
        batch_label = batch_label

        batch_pred = model_forward(batch_input)
        if isinstance(batch_pred, list):
            batch_pred = torch.tensor(batch_pred)

        recorder['loss'].append(
            loss_fn(batch_pred.to('cpu'), batch_label.to('cpu')))
        prec1, prec5 = accuracy(torch.tensor(batch_pred).to(
            'cpu'), batch_label.to('cpu'), topk=(1, 5))
        recorder['top1_accuracy'].append(prec1.item())
        recorder['top5_accuracy'].append(prec5.item())

    print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'.format(
            top1=sum(recorder['top1_accuracy'])/len(recorder['top1_accuracy']),
            top5=sum(recorder['top5_accuracy'])/len(recorder['top5_accuracy'])))

    return sum(recorder['top1_accuracy'])/len(recorder['top1_accuracy'])
