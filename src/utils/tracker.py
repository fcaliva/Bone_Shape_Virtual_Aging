import numpy as np

class tracker_recon_segment(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.stored_loss = np.empty((1))
        self.stored_score = np.empty((1))
        self.stored_rmse = np.empty((1))
        self.stored_mae = np.empty((1))
        self.stored_mse = np.empty((1))
        self.stored_nrmse = np.empty((1))
        self.stored_ssim = np.empty((1))
        self.cum_loss = 0.0
        self.cum_score = 0.0
        self.cum_rmse = 0.0
        self.cum_mae = 0.0
        self.cum_mse = 0.0
        self.cum_nrmse = 0.0
        self.cum_ssim = 0.0
        self.iter = 0


    def increment(self, loss, score, rmse, mae, mse, nrmse, ssim):
        if self.iter > 0:
            self.stored_loss = np.append(self.stored_loss, loss)
            self.stored_score = np.append(self.stored_score, score)
            self.stored_rmse = np.append(self.stored_rmse, rmse)
            self.stored_mae = np.append(self.stored_mae, mae)
            self.stored_mse = np.append(self.stored_mse, mse)
            self.stored_nrmse = np.append(self.stored_mse, nrmse)
            self.stored_ssim = np.append(self.stored_ssim, ssim)
        else:
            self.stored_loss = loss
            self.stored_score = score
            self.stored_rmse  = rmse
            self.stored_mae = mae
            self.stored_mse = mse
            self.stored_nrmse = nrmse
            self.stored_ssim = ssim

        self.cum_loss += loss
        self.cum_score += score
        self.cum_rmse += rmse
        self.cum_mae += mae
        self.cum_mse += mse
        self.cum_nrmse += nrmse
        self.cum_ssim += ssim
        self.iter += 1

    def average(self):
        return self.cum_loss/self.iter, self.cum_score/self.iter, self.cum_rmse/self.iter, self.cum_mae/self.iter, self.cum_mse/self.iter, self.cum_nrmse/self.iter, self.cum_ssim/self.iter

    def stdev(self):
        return np.std(self.stored_loss), np.std(self.stored_score), np.std(self.stored_rmse), np.std(self.stored_mae), np.std(self.stored_mse), np.std(self.stored_nrmse), np.std(self.stored_ssim)
