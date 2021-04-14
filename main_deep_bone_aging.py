# DioscoriDESS was Greek Physician who worked in Rome in the 1st Century A.D.
# author of De Materia Medica. He recommended to use ivy for treatment of OA.
# main codebase written by Claudia Iriondo and Francesco Calivà, email us with any questions:
# Email: francesco.caliva@ucsf.edu
# Email: claudia.iriondo@ucsf.edu
# If you use this code please cite:
#
#
#
#
from scipy.io import savemat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./utils')
import pdb 
import argparse
import yaml
import dataLoader as dataLoader
from logger import logger
from pprint import pprint
from utils import losses, early_stop, metrics, tb_vis, optimizers, save_figure
from utils.tracker import tracker_recon_segment as tracker


trial = False
if not trial:
    parser = argparse.ArgumentParser(description='define configuration file and run description')
    parser.add_argument('--cfg')
    parser.add_argument('--desc')
    args = parser.parse_args()
    with open(args.cfg) as f:
         config = yaml.load(f, Loader=yaml.UnsafeLoader)
    desc = args.desc
else:
    yaml_path = 'cfgs/train.yaml'
    desc = 'virtual_bone_shape_aging'
    with open(yaml_path) as f:
        config = yaml.load(f)
sys.path.append('./models/'+config['model_folder'])
import network as nn

if not os.path.exists(config['common']['log_path']):
    os.makedirs(config['common']['log_path'])
if not os.path.exists(config['common']['save_path']):
    os.makedirs(config['common']['save_path'])

sys.stdout = logger(sys.stdout,path=config['common']['log_path'],desc=desc)
print('\n\n',sys.stdout.name,'\n\n')
pprint(config)

if 'all' not in config['common']['vis_GPU'] or config['common']['qsub']!=1:
    os.environ['CUDA_VISIBLE_DEVICES'] = config['common']['vis_GPU']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
c = tf.ConfigProto()
#c.gpu_options.visible_device_list = "0"#GPU_VIS
c.gpu_options.allow_growth=True
c.gpu_options.per_process_gpu_memory_fraction = 0.95
c.allow_soft_placement = True
c.log_device_placement = False

seed = config['common']['seed']
np.random.seed(seed)
tf.reset_default_graph()
tf.set_random_seed(seed)

model = nn.__dict__[config['model']](**config['model_params'])

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)

undersampled_shape = np.concatenate(([config['data_train']['batch_size']],config['data_train']['im_dims'],[config['data_train']['num_channels']]))
fullysampled_shape = np.concatenate(([config['data_train']['batch_size']], config['data_train']['im_dims'], [config['data_train']['num_classes']]))

undersampled = tf.placeholder(dtype=tf.float32, shape=undersampled_shape)
fullysampled = tf.placeholder(dtype=tf.float32, shape=fullysampled_shape)

keep_prob = tf.placeholder(dtype=tf.float32, shape=())
loader_train = dataLoader.__dict__[config['learn']['dataloader']](**config['data_train'])
loader_val = dataLoader.__dict__[config['learn']['dataloader']](**config['data_val'])

logits_reconstruction = model.network_fn(undersampled,keep_prob)

weights = tf.constant([config['learn']['weights']], dtype='float32')


optimizer = optimizers.__dict__[config['learn']['optimizer']](config['learn']['lr'],global_step)

loss, pred_reconstruction = losses.__dict__[config['learn']['loss']](fullysampled, logits_reconstruction, weights)


one_over_mae_score, mae_score, mse_score, rmse_score, ssim_score = metrics.__dict__[config['learn']['metrics']](fullysampled, logits_reconstruction)

if config['learn']['monitor']=='mae':
    monitored_score = one_over_mae_score
    metric_name = '1/MAE'
    print(f'>>>> Monitoring Validation on {metric_name} metric <<<<')

trainer = optimizer.minimize(loss = loss, global_step = global_step)

train_summary_op = tb_vis.summarize2D_recon(config['learn']['comp'], 'train', undersampled, fullysampled, logits_reconstruction, config['data_train']['im_dims'][0], config['data_train']['im_dims'][1], mae_score, mse_score, ssim_score)
val_summary_op = tb_vis.summarize2D_recon(config['learn']['comp'], 'val',  undersampled, fullysampled, logits_reconstruction,
                                          config['data_val']['im_dims'][0],  config['data_val']['im_dims'][1], mae_score, mse_score, ssim_score)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

saver = tf.train.Saver(max_to_keep=15)

patience = early_stop.early_stop(patience=config['learn']['patience'])

if config['pretrain']['only_infer']:
    save_inference_in_path = f'{config["common"]["inference_path"]}{sys.stdout.name}'
    if not os.path.isdir(save_inference_in_path):
        os.makedirs(save_inference_in_path)

save_visuals_in_path = f'{config["common"]["visual_path"]}{sys.stdout.name}'
if not os.path.isdir(save_visuals_in_path):
    os.makedirs(save_visuals_in_path)        

print('\n\n',sys.stdout.name,'\n\n')
with tf.Session(config=c) as sess:
    writer = tf.summary.FileWriter(config['common']['save_path']+sys.stdout.name,sess.graph)
    init_op.run()
    if config['pretrain']['flag']:
        model_name = config['pretrain']['ckpt']
        if "model.ckpt-" not in model_name:
            ckpt_id_available = ([x.split('model.ckpt-')[-1].split('.')[0] for x in os.listdir(model_name)])
            ckpt_id_available = np.max([np.int(x) for x in ckpt_id_available if(x!='events' and x!='checkpoint')])
            model_name = model_name+ '/model.ckpt-'+str(ckpt_id_available)
        print(f'Running inference on {model_name}')
        
        if config['pretrain']['only_infer']:
            print(f'Output in {save_inference_in_path}')
        saver.restore(sess, model_name)
        
        val_track = tracker(num_classes=config['data_val']['num_classes'])
        for viter in range(np.max([np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int), 100])):
            try:
                [undersampled_val, fullysampled_val, name_val] = loader_val.fetch_batch()
            except:
                print('fetch_batch on val set did not work')
                loader_val.batch_cnt += 1
                continue

            val_summary, val_loss, val_recon, val_score, val_mae, val_mse, val_rmse, val_ssim = sess.run([val_summary_op, loss, pred_reconstruction, monitored_score, mae_score, mse_score, rmse_score, ssim_score], feed_dict={
                undersampled: undersampled_val, fullysampled: fullysampled_val, keep_prob: 1.0})

            val_nrmse = np.sqrt(
                np.power(fullysampled_val-val_recon, 2).mean())/fullysampled_val.mean()
            val_track.increment(
                val_loss, val_score, val_rmse, val_mae, val_mse, val_nrmse, 1.0-val_ssim)
            
            if config['pretrain']['only_infer']:
                for bb in range(config['data_val']['batch_size']):                    
                    savemat(f'{save_inference_in_path}/{name_val[bb].split("/")[-1].split(".")[0]}_pred.mat', {'input': np.squeeze(undersampled_val[bb]), 'target': np.squeeze(fullysampled_val[bb]), 'prediction': np.squeeze(val_recon[bb])})
                
        val_cum_loss, val_cum_score, val_rmse_score_mean, val_mae_score_mean, val_mse_score_mean, val_nrmse_score_mean, val_ssim_score_mean = val_track.average()
        val_cum_loss_std, val_cum_score_std, val_rmse_score_std, val_mae_score_std, val_mse_score_std, val_nrmse_score_std, val_ssim_score_std = val_track.stdev()

        if config['pretrain']['only_infer']:
            print(f'Inference loss: {val_cum_loss:.4f}({val_cum_loss_std:.4f}),\t Monitored Metric: {val_cum_score:.4f}({val_cum_score_std:.4f}),\t MAE: {val_mae_score_mean: .4f}({val_mae_score_std: .4f}), \t SSIM: {val_ssim_score_mean: .4f}({val_ssim_score_std: .4f}), \t MSE: {val_mse_score_mean: .4f}({val_mse_score_std: .4f}), \t NRMSE: {val_nrmse_score_mean: .4f}({val_nrmse_score_std: .4f}), \t RMSE: {val_rmse_score_mean: .4f}({val_rmse_score_std: .4f})')
            save_figure(undersampled_val, fullysampled_val, val_recon, save_visuals_in_path, 'val')
            print('Inference complete')
            exit()            
        else:
            print(f'(Restored) Validation loss: {val_cum_loss:.4f}({val_cum_loss_std:.4f}),\t Monitored Metric: {val_cum_score:.4f}({val_cum_score_std:.4f}),\t MAE: {val_mae_score_mean: .4f}({val_mae_score_std: .4f}), \t SSIM: {val_ssim_score_mean: .4f}({val_ssim_score_std: .4f}), \t MSE: {val_mse_score_mean: .4f}({val_mse_score_std: .4f}), \t NRMSE: {val_nrmse_score_mean: .4f}({val_nrmse_score_std: .4f}), \t RMSE: {val_rmse_score_mean: .4f}({val_rmse_score_std: .4f})')
            _, _ = patience.track(val_cum_score)
            print('Restored successfully')

    print('Training...')
    train_track = tracker(num_classes=config['data_train']['num_classes'])
    for iter in range(config['learn']['max_steps']):
        try:
            [ undersampled_train, fullysampled_train, name_train ] = loader_train.fetch_batch()
        except:
            print('fetch_batch on train set did not work')
            loader_train.batch_cnt +=1
            continue
        _, summary, current_loss, model_reconstruction, train_score, train_mae_score, train_mse_score, train_rmse_score, train_ssim = sess.run([trainer, train_summary_op, loss, pred_reconstruction, monitored_score, mae_score, mse_score, rmse_score, ssim_score], feed_dict={undersampled: undersampled_train, fullysampled: fullysampled_train, keep_prob: config['learn']['keep_prob']})        

        train_nrmse = np.sqrt(np.power(fullysampled_train-model_reconstruction,2).mean())/fullysampled_train.mean()
        train_track.increment(current_loss, train_score, train_rmse_score, train_mae_score, train_mse_score, train_nrmse, 1.0-train_ssim)
        
        if iter == 0:
            save_figure(undersampled_train, fullysampled_train,
                        model_reconstruction, save_visuals_in_path,'beginning')
        if iter != 0 and iter %config['common']['print_freq']==0:
            train_cum_loss,     train_cum_score,     train_rmse_score_mean, train_mae_score_mean, train_mse_score_mean, train_nrmse_score_mean, train_ssim_score_mean  = train_track.average()
            train_cum_loss_std, train_cum_score_std, train_rmse_score_std, train_mae_score_std,     train_mse_score_std, train_nrmse_score_std, train_ssim_score_std = train_track.stdev()
            
            print(f'Iteration: {iter},\t Training loss: {train_cum_loss:.4f}({train_cum_loss_std:.4f}),\t Training Monitored Metric: {train_cum_score:.4f}({train_cum_score_std:.4f}),\t MAE: {train_mae_score_mean:.4f}({train_mae_score_std:.4f}),\t SSIM: {train_ssim_score_mean:.4f}({train_ssim_score_std:.4f}), \t MSE: {train_mse_score_mean: .4f}({train_mse_score_std: .4f}), \t NRMSE: {train_nrmse_score_mean: .4f}({train_nrmse_score_std: .4f}), \t RMSE: {train_rmse_score_mean: .4f}({train_rmse_score_std: .4f})')
            save_figure(undersampled_train, fullysampled_train, model_reconstruction,save_visuals_in_path,'train')
            writer.add_summary(summary, sess.run(global_step))
            writer.flush()
            train_track = tracker(num_classes=config['data_train']['num_classes'])

        if iter != 0 and iter %config['learn']['val_freq']==0:            
            val_track = tracker(num_classes=config['data_val']['num_classes'])

            for viter in range(np.min([np.floor(loader_val.__len__()/config['data_val']['batch_size']).astype(int), 100])):
                try:
                    [undersampled_val, fullysampled_val,
                        name_val] = loader_val.fetch_batch()
                except:
                    print('fetch_batch on val set did not work')
                    loader_val.batch_cnt += 1
                    continue                
                val_summary, val_loss, val_recon, val_score, val_mae, val_mse, val_rmse, val_ssim = sess.run([val_summary_op, loss, pred_reconstruction, monitored_score, mae_score, mse_score, rmse_score, ssim_score], feed_dict={
                                                                                                                    undersampled: undersampled_val, fullysampled: fullysampled_val, keep_prob: 1.0})

                val_nrmse = np.sqrt(np.power(fullysampled_val-val_recon,2).mean())/fullysampled_val.mean()
                val_track.increment(val_loss,val_score, val_rmse, val_mae, val_mse, val_nrmse, 1.0-val_ssim)
                
            val_cum_loss, val_cum_score, val_rmse_score_mean, val_mae_score_mean, val_mse_score_mean, val_nrmse_score_mean, val_ssim_score_mean = val_track.average()
            val_cum_loss_std, val_cum_score_std, val_rmse_score_std, val_mae_score_std, val_mse_score_std, val_nrmse_score_std, val_ssim_score_std = val_track.stdev()

            print(f'Validation loss: {val_cum_loss:.4f}({val_cum_loss_std:.4f}),\t Monitored Metric: {val_cum_score:.4f}({val_cum_score_std:.4f}),\t MAE: {val_mae_score_mean: .4f}({val_mae_score_std: .4f}), \t SSIM: {val_ssim_score_mean: .4f}({val_ssim_score_std: .4f}), \t MSE: {val_mse_score_mean: .4f}({val_mse_score_std: .4f}), \t NRMSE: {val_nrmse_score_mean: .4f}({val_nrmse_score_std: .4f}), \t RMSE: {val_rmse_score_mean: .4f}({val_rmse_score_std: .4f})')
            save_figure(undersampled_val, fullysampled_val, val_recon,save_visuals_in_path, 'val')

            writer.add_summary(val_summary, sess.run(global_step))
            writer.flush()

            save_flag, stop_flag = patience.track(val_cum_score)

            if save_flag:
                print(f'>>>!!!New checkpoint at step: {iter}\t, Validation Score: {val_cum_score:.4f}\t')
                checkpoint_path = config['common']['save_path']+sys.stdout.name+'/model.ckpt'
                saver.save(sess, checkpoint_path, global_step = global_step)
                save_figure(undersampled_val,
                            fullysampled_val, val_recon,save_visuals_in_path, 'val-best')
            if stop_flag:
                print('Stopping model due to no improvement for {} validation runs'.format(config['learn']['patience']) )
                writer.close()
                sess.close()
                break

    writer.close()
    print('Model finished training for {} steps'.format(iter))
