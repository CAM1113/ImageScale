from torch import nn
import torch


class ScaleLoss(nn.Module):
    def __init__(self):
        super(ScaleLoss, self).__init__()

    def forward(self, pred, enlarge_images, target_image):
        batch_num = target_image.shape[0]
        pred_image = enlarge_images + pred
        pred_image = pred_image.view(batch_num, -1)
        target_image = target_image.view(batch_num, -1)
        loss = torch.sum((target_image - pred_image) ** 2) / batch_num
        return loss
