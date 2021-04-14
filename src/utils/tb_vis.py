
import tensorflow as tf
import pdb

def summarize2D_recon(comp, type, input, gt, pred, r, c, loss_mae, loss_mse, ssim_score):
    input = tf.reshape(input[0, :, :, 0], [-1, r, c, 1])
    target = tf.reshape(gt[0, :, :, 0], [-1, r, c, 1])
    prediction = tf.reshape(pred[0, :, :, 0], [-1, r, c, 1])

    tb_summary = {}
    tb_summary['input'] = tf.summary.image(type+'/input', input)
    tb_summary['target'] = tf.summary.image(type+'/fullysampled', target)
    tb_summary['prediction'] = tf.summary.image(type+'/prediction', prediction)
    tb_summary['mae'] = tf.summary.scalar(type+'/mae', loss_mae)
    tb_summary['mse'] = tf.summary.scalar(type+'/mse', loss_mse)
    tb_summary['ssim'] = tf.summary.scalar(type+'/ssim', ssim_score)

    summary_op = tf.summary.merge(list(tb_summary.values()))
    return summary_op
