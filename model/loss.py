import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    criterion = nn.MSELoss(reduction='mean')
    return criterion(output, target)

def bce_loss(output, target):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    return criterion(output, target)

def loss_fn(lbl, y):
    """ loss function between true labels lbl and prediction y """
    # prediction y: [bz, 3, w, h], flow_y, flow_x, cell prob
    # label lbl: [bz, 3, w, h], flow_y, flow_x, cell_prob
    # flows
    flows = 5. * lbl[:, :2]
    # prob
    prob = lbl[:, 2].abs()

    loss = mse_loss(y[:, :2], flows)
    loss2 = bce_loss(y[:, 2], prob)
    loss = loss / 2. + loss2
    return loss