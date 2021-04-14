from utils.losses import SSIM_loss, mae_loss, mse_loss
import tensorflow as tf
import numpy as np

def structuralSimilarity(gt_ssim, pred_ssim):
    [ssim_score, ssim_map] = SSIM_loss(gt_ssim, pred_ssim)
    return ssim_score


def mae_metric(gt, pred):
    axis = np.arange(0, len(pred.get_shape().as_list())-1).tolist()
    error = tf.cast(gt, tf.float32) - pred
    absError = tf.abs(error)
    mae_classes = tf.reduce_mean(absError, axis=axis)
    mae_score = tf.reduce_mean(mae_classes)
    return mae_score, 1.0/mae_score


def mse_metric(gt, pred):
    axis = np.arange(0, len(pred.get_shape().as_list())-1).tolist()
    error = tf.cast(gt, tf.float32) - pred
    sqError = tf.pow(error, 2)

    mse_classes = tf.reduce_mean(sqError, axis=axis)
    mse_score = tf.reduce_mean(mse_classes)

    return mse_score, 1.0/mse_score


def rmse_metric(gt, pred):
    axis = np.arange(0, len(pred.get_shape().as_list())-1).tolist()
    error = tf.cast(gt, tf.float32) - pred
    sqError = tf.pow(error, 2)

    mse_classes = tf.reduce_mean(sqError, axis=axis)
    rmse_classes = tf.sqrt(mse_classes)
    rmse_score = tf.reduce_mean(rmse_classes)
    return rmse_score, 1.0/rmse_score


def mae_mse_rmse_ssim(gt,pred):
    
    mae_score, one_over_mae_score = mae_metric(gt, pred)
    mse_score, one_over_mae_mse_score = mse_metric(gt, pred)
    rmse_score, one_over_mae_rmse_score = rmse_metric(gt, pred)
    try:
        ssim_score = structuralSimilarity(gt, pred)
    except:
        ssim_score, _ = mae_metric(gt, pred)
    return one_over_mae_score, mae_score, mse_score, rmse_score, ssim_score
