import numpy as np
import torch
import skimage.measure
import skimage.metrics
from utils import geo_utils


def compute_img_metric(rgb, gt):
    '''
    rgb and gt: in shape [h,w,3]
    '''
    ssim = skimage.metrics.structural_similarity(gt, rgb, data_range=1, channel_axis=-1)
    psnr = skimage.metrics.peak_signal_noise_ratio(gt, rgb, data_range=1)
    return psnr, ssim


def compute_pose_metric(pred, gt):
    '''
    https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py
    pred, gt in shape [7]
    '''
    q_pred = pred[:4].numpy()
    q = gt[:4].numpy()
    d = np.abs(np.sum(np.multiply(q_pred, q)))
    theta = 2 * np.arccos(d) * 180 / np.pi

    t_pred = pred[4:].numpy()
    t = gt[4:].numpy()
    t_error = np.linalg.norm(t_pred - t)
    return theta, t_error