import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import functional as F



def depth_to_xyz(depthImage, dataset_name):
    ## input depth image[B, 1, H, W]
    ## output xyz image[B, 3, H, W]
    if dataset_name == 'syntodd':
        fx = 613.9624633789062
        fy = 613.9624633789062
        du = 320.0
        dv = 240.0
    elif dataset_name == 'clearpose':
        fx = 601.46000163
        fy = 601.5933431
        du = 334.89998372
        dv = 248.15334066
    else:
        raise ValueError("Dataset name not correct !")

    B, C, H, W = depthImage.shape
    device = depthImage.device

    xyz = torch.zeros([B, H, W, 3], device=device)
    imageIndexX = torch.arange(0, W, 1, device=device) - du
    imageIndexY = torch.arange(0, H, 1, device=device) - dv
    depthImage = depthImage.squeeze()
    if B == 1:
        depthImage = depthImage.unsqueeze(0)

    xyz[:, :, :, 0] = depthImage/fx * imageIndexX
    xyz[:, :, :, 1] = (depthImage.transpose(1, 2)/fy * imageIndexY.T).transpose(1, 2)
    xyz[:, :, :, 2] = depthImage
    xyz = xyz.permute(0, 3, 1, 2).to(device)
    return xyz


def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def get_surface_normal(x, dataset_name):
    xyz = depth_to_xyz(x, dataset_name)
    dx,dy = gradient(xyz)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal, dim=1, keepdim=True)+1e-8)
    return surface_normal, dx, dy


def visualize_surface_normals(normals):
    normals = (normals + 1) / 2

    normal_image = normals[0].permute(1, 2, 0).cpu().numpy()

    plt.imshow(normal_image)
    plt.title("Surface Normals")
    plt.axis('off')
    plt.show()


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        edge_ky = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = Sobel().cuda()

    def forward(self, input, target, zero_mask=None, loss_mask=None):

        input_grad = self.sobel(input)
        target_grad = self.sobel(target)

        input_grad_dx = input_grad[:, 0, :, :].contiguous().view_as(input)
        input_grad_dy = input_grad[:, 1, :, :].contiguous().view_as(input)
        target_grad_dx = target_grad[:, 0, :, :].contiguous().view_as(target)
        target_grad_dy = target_grad[:, 1, :, :].contiguous().view_as(target)

        loss_dx = torch.abs(input_grad_dx - target_grad_dx)
        loss_dy = torch.abs(input_grad_dy - target_grad_dy)

        if zero_mask is None:
            zero_mask = torch.ones_like(loss_dx, dtype=torch.bool).to(loss_dx.device)
        else:
            zero_mask = zero_mask.unsqueeze(1).expand_as(loss_dx)

        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(1).expand_as(loss_dx)
            return loss_dx[loss_mask].mean() + loss_dy[loss_mask].mean() + 0.01 * (loss_dx[zero_mask].mean() + loss_dy[zero_mask].mean())

        # return loss_dx[zero_mask].mean() + loss_dy[zero_mask].mean()
        return loss_dx.mean() + loss_dy.mean()


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, normal_pred, normal_gt, zero_mask=None, loss_mask=None):
        loss = self.loss(normal_pred, normal_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool).to(loss.device)
        else:
            zero_mask = zero_mask.unsqueeze(1).expand_as(normal_gt)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(1).expand_as(normal_gt)
            return loss[loss_mask].mean() + 0.01 * loss[zero_mask].mean()
        # return loss[zero_mask].mean() 
        return loss.mean() 


class SegIouLoss(nn.Module):
    def __init__(self, num_classes):
        super(SegIouLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, seg_pred, seg_gt):
        seg_pred_mask = F.softmax(seg_pred, dim=1)
        assert seg_pred_mask.shape[2] == seg_gt.shape[1] and seg_pred_mask.shape[3] == seg_gt.shape[2]
        IoUs = []
        for cls in range(self.num_classes):
            pred_cls = seg_pred_mask[:, cls, :, :]
            target_cls = (seg_gt == cls).float()

            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) - intersection

            iou = (intersection + 1e-6) / (union + 1e-6)
            IoUs.append(iou)

        mean_iou = torch.mean(torch.stack(IoUs, dim=0), dim=0)
        loss = 1 - mean_iou.mean()
        
        return loss


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None):

        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool).to(loss.device)
        if loss_mask is not None:
            return loss[loss_mask].mean() + 0.01 * loss[zero_mask].mean()
        # return loss[zero_mask].mean()
        return loss.mean()


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None, ):

        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool).to(loss.device)
        if loss_mask is not None:
            return loss[loss_mask].mean() + 0.01 * loss[zero_mask].mean()
        return loss[zero_mask].mean()


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super(MaskedSmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, depth_pred, depth_gt, zero_mask=None, loss_mask=None):

        loss = self.loss(depth_pred, depth_gt)
        if zero_mask is None:
            zero_mask = torch.ones_like(loss, dtype=torch.bool).to(loss.device)
        if loss_mask is not None:
            return loss[loss_mask].mean() + 0.01 * loss[zero_mask].mean()
        return loss[zero_mask].mean()    