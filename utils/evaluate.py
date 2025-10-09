import numpy as np
import torch

from torch.nn import functional as F
import torch.nn as nn

DEPTH_MAX_STNTODD = 3.0
DEPTH_MIN_STNTODD = 0.0

DEPTH_MAX_TRANSCG = 1.0
DEPTH_MIN_TRANSCG = 0.3


def compute_depth_metrics(preds, gts, depth_min, depth_max, zero_masks=None, denorm=True, gt_masks=None, num_classes=2):
    eps=1e-5
    
    preds = preds.clone()
    gts = gts.clone()
    if denorm:
        preds = preds * (depth_max - depth_min) + depth_min
        gts = gts * (depth_max - depth_min) + depth_min
    scale_factor = float(gts.shape[2]) / float(preds.shape[2])
    if scale_factor != 1.0:
        gts = F.interpolate(
            gts[:, None, :, :], scale_factor=scale_factor
        )[:, 0, :, :]

    if zero_masks is None:
        zero_masks = (preds > eps).to(preds.device)
    
    RMSE = computeRMSE(preds, gts, zero_masks)
    MAE = computeMAE(preds, gts, zero_masks)
    REL = computeREL(preds, gts, zero_masks)
    DELTA105, DELTA110, DELTA125 = computeDelta(preds, gts, zero_masks)
   
    masks = (gt_masks == (num_classes - 1))
    masks = masks & zero_masks
    
    RMSE_ = computeRMSE(preds, gts, masks)
    MAE_ = computeMAE(preds, gts, masks)
    REL_ = computeREL(preds, gts, masks)
    DELTA105_, DELTA110_, DELTA125_ = computeDelta(preds, gts, masks)

    return MAE, RMSE, REL, DELTA105, DELTA110, DELTA125, MAE_, RMSE_, REL_, DELTA105_, DELTA110_, DELTA125_
    

def computeRMSE(pred, gt, mask):
    eps=1e-5
    result = torch.sum(((pred - gt) ** 2) * mask.float(), dim=[1, 2]) / (torch.sum(mask.float(), dim=[1, 2]) + eps)
    return torch.mean(torch.sqrt(result)).item()                                                                                                                                                                                 

def computeMAE(pred, gt, mask):
    eps=1e-5
    result = torch.sum(torch.abs(pred - gt) * mask.float(), dim=[1, 2]) / (torch.sum(mask.float(), dim=[1, 2]) + eps)
    return torch.mean(result).item()

def computeREL(pred, gt, mask):
    eps = 1e-5
    result = torch.sum((torch.abs(pred - gt) / (gt + eps)) * mask.float(), dim=[1, 2]) / (torch.sum(mask.float(), dim=[1, 2]) + eps)
    return torch.mean(result).item()

def computeDelta(pred, gt, mask):
    result = []
    eps = 1e-5
    deltas = [1.05, 1.10, 1.25]
    thres = torch.maximum(pred / (gt + eps), gt / (pred + eps))
    for delta in deltas:
        res = ((thres < delta) & mask).float().sum(dim=[1, 2]) / (torch.sum(mask.float(), dim=[1, 2]) + eps)
        result.append(torch.mean(res).item() * 100)
    return result


def compute_seg_metrics(preds, gts, threshold=0.5,  num_classes=3):

    if not isinstance(preds, np.ndarray):
        preds = np.ascontiguousarray(preds.detach().cpu().numpy())
    if not isinstance(gts, np.ndarray):
        gts = np.ascontiguousarray(gts.detach().cpu().numpy())

    iou_per_class_total = np.zeros(num_classes)
    ap_per_class_total = np.zeros(num_classes)
    batch_size = preds.shape[0]

    for pred, gt in zip(preds, gts):
        pred = np.argmax(pred, axis=0)
        iou_per_class = computeIoU(pred, gt, num_classes)
        iou_per_class_total += np.array(iou_per_class)
        ap_per_class_total += np.array([1 if iou > threshold else 0 for iou in iou_per_class])

    iou_per_class_avg = iou_per_class_total / batch_size
    mAP = np.mean(ap_per_class_total / batch_size)
    IoU = np.mean(iou_per_class_avg)

    return IoU, mAP


def computeIoU(pred, gt, num_classes):
    iou_per_class = []
    assert pred.shape == gt.shape
    smooth = 1e-6

    for cls in range(num_classes):
        pred_class = (pred == cls)
        gt_class = (gt == cls)
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou)

    return iou_per_class
