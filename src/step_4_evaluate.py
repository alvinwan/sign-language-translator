from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import onnx
import onnxruntime as ort

from step_2_dataset import get_train_test_loaders
from step_3_train import Net


def evaluate(outputs: Variable, labels: Variable, normalized: bool=True
             ) -> float:
    """Evaluate neural network outputs against non-one-hotted labels."""
    Y = labels.data.numpy()
    Yhat = np.argmax(outputs, axis=1)
    denom = Y.shape[0] if normalized else 1
    return float(np.sum(Yhat == Y) / denom)


def batch_evaluate(
        net: Net,
        dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = n = 0.0
    for batch in dataloader:
        n += batch['image'].size(0)
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.data.numpy()
        score += evaluate(outputs, batch['label'][:, 0], False)
    return score / n


def validate():
    trainloader, testloader = get_train_test_loaders()
    net = Net().float()

    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    print('=' * 10, 'PyTorch', '=' * 10)
    train_acc = batch_evaluate(net, trainloader)
    print('Training accuracy: %.3f' % train_acc)
    test_acc = batch_evaluate(net, testloader)
    print('Validation accuracy: %.3f' % test_acc)

    trainloader, testloader = get_train_test_loaders(1)

    # export to onnx
    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    # check exported model
    model = onnx.load(fname)
    onnx.checker.check_model(model)  # check model is well-formed

    # create runnable session with exported model
    ort_session = ort.InferenceSession(fname)
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(net, trainloader)
    print('Training accuracy: %.3f' % train_acc)
    test_acc = batch_evaluate(net, testloader)
    print('Validation accuracy: %.3f' % test_acc)


if __name__ == '__main__':
    validate()
