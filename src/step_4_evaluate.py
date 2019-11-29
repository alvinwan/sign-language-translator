from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

from step_2_dataset import get_train_test_loaders
from step_3_train import Net


def evaluate(outputs: Variable, labels: Variable, normalized: bool=True
             ) -> float:
    """Evaluate neural network outputs against non-one-hotted labels."""
    Y = labels.data.numpy()
    Yhat = np.argmax(outputs.data.numpy(), axis=1)
    denom = Y.shape[0] if normalized else 1
    return float(np.sum(Yhat == Y) / denom)


def batch_evaluate(
        net: Net,
        dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = n = 0.0
    for batch in dataloader:
        n += batch['image'].size(0)
        score += evaluate(net(batch['image']), batch['label'][:, 0], False)
    return score / n


def validate():
    trainloader, testloader = get_train_test_loaders()
    net = Net().float()

    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)

    train_acc = batch_evaluate(net, trainloader)
    print('Training accuracy: %.3f' % train_acc)
    test_acc = batch_evaluate(net, testloader)
    print('Validation accuracy: %.3f' % test_acc)


if __name__ == '__main__':
    validate()
