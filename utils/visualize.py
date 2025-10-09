import cv2
import colour
import torch
import numpy as np
import matplotlib.pyplot as plt


def vis_error_map(depth_pred, depth_gt, type):
    colormap = plt.cm.Reds
    if type == 'depth':
        if not isinstance(depth_pred, np.ndarray):
            depth_pred = np.ascontiguousarray(depth_pred.detach().cpu().numpy())
        if not isinstance(depth_gt, np.ndarray):
            depth_gt = np.ascontiguousarray(depth_gt.detach().cpu().numpy())
        error_gt_visual = np.abs(depth_gt - depth_pred)
        # error_map_gt = 1.0 - error_map_gt
        error_gt_visual = (error_gt_visual * 255).astype(np.uint8)
        error_gt_visual = colormap(error_gt_visual)
        error_gt_visual = (error_gt_visual[:, :, :3] * 255).astype(np.uint8)
        return error_gt_visual 

def vis_depth(depth):
    colormap = plt.cm.YlGnBu
    if not isinstance(depth, np.ndarray):
        depth_visual = np.ascontiguousarray(depth.detach().cpu().numpy())

    assert len(depth.shape) == 2
    depth_visual = depth_visual.astype(np.float64)
    # depth_visual = 1. - depth_visual
    depth_visual = (depth_visual * 255).astype(np.uint8)
    depth_visual = colormap(depth_visual)
    depth_visual = (depth_visual[:, :, :3] * 255).astype(np.uint8)
    return depth_visual


def get_syntodd_colors():
  colors = [
      colour.Color("yellow"),
      colour.Color("blue"),
      colour.Color("green"),
      colour.Color("red"),
      colour.Color("purple")
  ]
  color_rgb = 255 * np.array([np.array(a.get_rgb()) for a in colors])
  color_rgb = [a.astype(np.int) for a in color_rgb]
  return color_rgb


def get_clearpose_colors():
  colors = [
      colour.Color("yellow"),
      colour.Color("purple")
  ]
  color_rgb = 255 * np.array([np.array(a.get_rgb()) for a in colors])
  color_rgb = [a.astype(np.int) for a in color_rgb]
  return color_rgb


def color_img_to_gray(image):
  gray_scale_img = np.zeros(image.shape)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for i in range(3):
    gray_scale_img[:, :, i] = img
  gray_scale_img[:, :, i] = img
  return gray_scale_img


def vis_seg_gt(seg, img, dataset_name):
    if not isinstance(seg, np.ndarray):
        seg_visual = np.ascontiguousarray(seg.detach().cpu().numpy())
    if dataset_name == "syntodd":
        colors = get_syntodd_colors()
        assert len(seg_visual.shape) == 2
        seg_visual = seg_visual.astype(np.uint8)
        color_img_truths = color_img_to_gray(np.copy(img))
        for ii, color in zip(range(3), colors):
            colored_mask_truths = np.zeros([seg_visual.shape[0], seg_visual.shape[1], 3])
            colored_mask_truths[seg_visual == ii, :] = color
            color_img_truths = cv2.addWeighted(
                color_img_truths.astype(np.uint8), 0.9, colored_mask_truths.astype(np.uint8), 0.4, 0
            )
        return cv2.cvtColor(color_img_truths, cv2.COLOR_BGR2RGB)
    elif dataset_name == "clearpose":
        colors = get_clearpose_colors()
        assert len(seg_visual.shape) == 2
        seg_visual = seg_visual.astype(np.uint8)
        color_img_truths = color_img_to_gray(np.copy(img))
        for ii, color in zip(range(2), colors):
            colored_mask_truths = np.zeros([seg_visual.shape[0], seg_visual.shape[1], 3])
            colored_mask_truths[seg_visual == ii, :] = color
            color_img_truths = cv2.addWeighted(
                color_img_truths.astype(np.uint8), 0.9, colored_mask_truths.astype(np.uint8), 0.4, 0
            )
        return cv2.cvtColor(color_img_truths, cv2.COLOR_BGR2RGB)
    else:
        seg_visual = ((seg_visual.astype(np.uint8)) * 255).astype(np.uint8)
        return seg_visual
    

def vis_seg(seg, img, dataset_name):
    if not isinstance(seg, np.ndarray):
        seg_visual = np.ascontiguousarray(seg.detach().cpu().numpy())
    if dataset_name == "syntodd":
        colors = get_syntodd_colors()
        assert len(seg_visual.shape) == 3
        seg_visual = np.argmax(seg_visual, axis=0)
        seg_visual = seg_visual.astype(np.uint8)
        color_img_preds = color_img_to_gray(np.copy(img))
        for ii, color in zip(range(3), colors):
            colored_mask_preds = np.zeros([seg_visual.shape[0], seg_visual.shape[1], 3])
            colored_mask_preds[seg_visual == ii, :] = color
            color_img_preds = cv2.addWeighted(
                color_img_preds.astype(np.uint8), 0.9, colored_mask_preds.astype(np.uint8), 0.4, 0
            )
        return cv2.cvtColor(color_img_preds, cv2.COLOR_BGR2RGB)
    elif dataset_name == "clearpose":
        colors = get_clearpose_colors()
        assert len(seg_visual.shape) == 3
        seg_visual = np.argmax(seg_visual, axis=0)
        seg_visual = seg_visual.astype(np.uint8)
        color_img_preds = color_img_to_gray(np.copy(img))
        for ii, color in zip(range(2), colors):
            colored_mask_preds = np.zeros([seg_visual.shape[0], seg_visual.shape[1], 3])
            colored_mask_preds[seg_visual == ii, :] = color
            color_img_preds = cv2.addWeighted(
                color_img_preds.astype(np.uint8), 0.9, colored_mask_preds.astype(np.uint8), 0.4, 0
            )
        return cv2.cvtColor(color_img_preds, cv2.COLOR_BGR2RGB)
    else:
        seg_visual = (np.argmax(seg_visual, axis=0).astype(np.uint8) * 255).astype(np.uint8)
        return seg_visual
    


def vis_rgb(rgb, dataset_name=None):
    if dataset_name is not None:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        rgb_np = np.ascontiguousarray(rgb.detach().cpu().numpy())
        rgb_np = rgb_np * std[:, None, None] + mean[:, None, None]
        rgb_np = rgb_np.transpose((1, 2, 0))
        rgb_visual = np.clip(rgb_np * 255.0, 0, 255)
    else:
        rgb_visual = np.clip(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255)
    
    return rgb_visual.astype(np.uint8)