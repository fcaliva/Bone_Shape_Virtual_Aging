import tensorflow as tf
import numpy as np

def mae_loss(gt, logits, weights=[]):
    with tf.variable_scope('mae_loss'):
        pred = logits
        error = tf.subtract(tf.cast(gt, tf.float32), logits)
        absError = tf.abs(error)
        mae_loss = tf.reduce_mean(absError)
    return mae_loss, pred


def mse_loss(gt, logits, weights=[]):
    with tf.variable_scope('mse_loss'):
        pred = logits
        error = tf.subtract(tf.cast(gt, tf.float32), logits)
        squaredError = tf.pow(error, 2)

        mse_loss = tf.reduce_mean(squaredError)
    return mse_loss, pred


def rmse_loss(gt, logits, weights=[]):
    with tf.variable_scope('rmse_loss'):
        mse_loss, pred = mse_loss(gt, logits)
        rmse_loss = tf.sqrt(mse_loss)
    return rmse_loss, pred


def SSIM_loss(gt, logits, weights=[], k1=0.01, k2=0.03, L=1, window_size=11, dimensions=2, return_map=True):    
    # https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
    # Image quality assessment: from error visibility to structural similarity
    def _tf_fspecial_gauss(size, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function"""
        assert dimensions >= 2, "SSIM must be calculated with dimensions >= 2!"
        slices = [slice(-size//2 + 1, size//2 + 1) for item in range(dimensions)]
        data = [tf.constant(item, dtype=tf.float32) for item in np.mgrid[slices]]
        numerator = data[0] ** 2
        for i in range(1, dimensions):
            numerator += data[i] ** 2
        g = tf.exp(-(numerator/(2.0*sigma**2)))
        g = tf.expand_dims(g, axis=-1)
        g = tf.expand_dims(g, axis=-1)
        # tile to have proper shape
        if dimensions == 3:
            g = tf.tile(g, [1, 1, 1, gt.shape[-1], gt.shape[-1]])
        elif dimensions == 2:
            g = tf.tile(g, [1, 1, gt.shape[-1], gt.shape[-1]])
        return g / tf.reduce_sum(g)
    with tf.variable_scope('SSIM_loss'):
        window = _tf_fspecial_gauss(window_size)
        if dimensions == 3:
            stridekernel = [1 ,1, 1, 1, 1]
            mu1 = tf.nn.conv3d(gt, window, strides = stridekernel, padding = 'VALID')
            mu2 = tf.nn.conv3d(logits, window, strides = stridekernel, padding = 'VALID')

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2

            mu1_mu2 = mu1 * mu2

            sigma1_sq = tf.nn.conv3d(gt*gt, window, strides = stridekernel, padding = 'VALID') - mu1_sq
            sigma2_sq = tf.nn.conv3d(logits*logits, window, strides = stridekernel, padding = 'VALID') - mu2_sq
            sigma1_2 = tf.nn.conv3d(gt*logits, window, strides = stridekernel, padding = 'VALID') - mu1_mu2

            c1 = (k1*L)**2
            c2 = (k2*L)**2

            ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
            ssim_loss = tf.subtract(1.0,tf.reduce_mean(ssim_map))
        elif dimensions ==2:
            stridekernel = [1 ,1, 1, 1]

            mu1 = tf.nn.conv2d(gt, window, strides = stridekernel, padding = 'VALID')
            mu2 = tf.nn.conv2d(logits, window, strides = stridekernel, padding = 'VALID')

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2

            mu1_mu2 = mu1 * mu2

            sigma1_sq = tf.nn.conv2d(gt*gt, window, strides = stridekernel, padding = 'VALID') - mu1_sq
            sigma2_sq = tf.nn.conv2d(logits*logits, window, strides = stridekernel, padding = 'VALID') - mu2_sq
            sigma1_2 = tf.nn.conv2d(gt*logits, window, strides = stridekernel, padding = 'VALID') - mu1_mu2

            c1 = (k1*L)**2
            c2 = (k2*L)**2

            ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
            ssim_loss = tf.subtract(1.0,tf.reduce_mean(ssim_map))
        # if return_map == True:
        return ssim_loss, ssim_map
        # else:
            # return ssim_loss


def SSIM_MAE_loss(gt,logits,weights=[],mae_weight=6.7):
    ssim, ssim_map = SSIM_loss(gt,logits)
    mae, pred = mae_loss(gt, logits)
    loss = ssim + mae_weight * mae
    return loss, pred
