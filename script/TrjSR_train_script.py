import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import math
import random
import pickle
import argparse
import higher

from config import Config as Config
from utilities import tool_funcs
from nn.TrjSR import TrjSR, input_processing
from utilities.strategy_all import load_trajs, build_training_samples
import numpy as np


trjsr_epochs = 5
trjsr_batch_size = 64
trjsr_learning_rate = 0.0005
trjsr_training_bad_patience = 5


train_split_ratio = 0.7
eval_split_ratio = 0.8
eval_num = 1000
test_num = 2000
counts = int(7000/trjsr_batch_size)
raw_inter_bad_patience = 20

iteration_counts = 200

class TrajSimiRegression(nn.Module):
    def __init__(self, nin):
        # nin = traj_emb_size 
        super(TrajSimiRegression, self).__init__()
        self.enc = nn.Sequential(nn.Linear(nin, nin),
                                nn.ReLU(),
                                nn.Linear(nin, nin))

    def forward(self, trajs):
        # trajs: [batch_size, emb_size]
        return F.normalize(self.enc(trajs), dim=1) #[batch_size, emb_size]

class FullModel(nn.Module):
    def __init__(self, encoder, regression):
        super().__init__()
        self.encoder = encoder
        self.regression = regression
    def forward(self, *args, **kwargs):
        embs = self.encoder(*args, **kwargs)
        return self.regression(embs)

class TrjSRTrainer:
    def __init__(self):
        super(TrjSRTrainer, self).__init__()
        self.device = torch.device('cuda:{}'.format(Config.gpu_id))
        lon_range = (Config.min_lon, Config.max_lon)
        lat_range = (Config.min_lat, Config.max_lat)
        self.bounding_box = ((lon_range[0], lat_range[0]), (lon_range[1], lat_range[1]))
        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_TrjSR_{}_best{}_{}_{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid, Config.strategy, Config.use_meta_learning)
        self.mcts = None
        self.last_sampled_indices = None
        self.model = TrjSR(lon_range, lat_range, Config.trjsr_imgsize_x_lr, 
                        Config.trjsr_imgsize_y_lr, Config.trjsr_pixelrange_lr,
                        Config.traj_embedding_dim)
        self.model.to(self.device)
        self.regression = TrajSimiRegression(Config.traj_embedding_dim)
        self.regression.to(self.device)

        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            N = len(self.dic_datasets['trains_merc'])  # number of training samples
            # define a learnable parameter alpha: shape=[N]
            alpha_init = getattr(Config, 'alpha_init', 1.0)
            alpha_init_tensor = torch.ones(N, dtype=torch.float, device=self.device) * alpha_init
            self.alpha = nn.Parameter(alpha_init_tensor)
            
            # optimizer for alpha (outer-most), used to update alpha
            alpha_lr = getattr(Config, 'alpha_lr', 1e-4)
            self.opt_alpha = torch.optim.Adam([self.alpha], lr=alpha_lr)

            # checkpoint = torch.load(self.checkpoint_path.replace('_True', '_False'))
            # self.model.load_state_dict(checkpoint['model'])
            # self.model.to(self.device)
            # self.regression.load_state_dict(checkpoint['regression'])
            # self.regression.to(self.device)
            
        else:
            self.alpha = None
            self.opt_alpha = None

    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        self.mcts_pre_trained = True
        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            checkpoint = torch.load(self.checkpoint_path.replace('_True', '_False'), map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.regression.load_state_dict(checkpoint['regression'])
            self.regression.to(self.device)

        inter_bad_patience = raw_inter_bad_patience

        self.full_model = FullModel(self.model, self.regression).to(self.device)
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.model.parameters()))+
                                    list(filter(lambda p: p.requires_grad, self.regression.parameters())), \
                                    lr = trjsr_learning_rate)

        best_hr_eval = 0.0
        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            eval_metrics = self.__test(self.dic_datasets['evals_img'], self.dic_datasets['evals_simi'], self.dic_datasets['max_distance'])
            logging.info("meta learning enabled")
            logging.info("initialize the best hr eval.  loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(*eval_metrics))
            # best_hr_eval = np.mean(eval_metrics[1:])
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = trjsr_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(trjsr_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []
            iter_bad_counter = 0
            inter_best_hr_eval = best_hr_eval

            self.model.train()
            self.regression.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_merc'], \
                                                            self.dic_datasets['max_distance'],\
                                                         last_sampled_indices=self.last_sampled_indices)):
                
                if batch == 'EVAL':
                    eval_metrics = self.__test(self.dic_datasets['evals_img'], self.dic_datasets['evals_simi'], self.dic_datasets['max_distance'])
                    logging.info("eval.     step={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(i_batch, *eval_metrics))
                    eval_hr_ep = np.mean(eval_metrics[1:])
                    # eval_loss_ep = eval_metrics[0]
                    if eval_hr_ep > inter_best_hr_eval:
                        iter_bad_counter = 0
                        inter_best_hr_eval = eval_hr_ep
                        torch.save({'model': self.model.state_dict(), 'regression': self.regression.state_dict()}, 
                           self.checkpoint_path)
                        continue
                    else:
                        iter_bad_counter += 1
                        if iter_bad_counter == inter_bad_patience:
                            logging.info("No improvement at step {}, early stopping.".format(i_batch))
                            # load the best model
                            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                            self.model.load_state_dict(checkpoint['model'])
                            self.model.to(self.device)
                            self.regression.load_state_dict(checkpoint['regression'])
                            self.regression.to(self.device)
                            self.mcts_pre_trained = False
                            break
                        else:
                            continue
                _time_batch = time.time()

                sub_trajs_img, sub_simi, sub_trajs_idxs = batch
                # sub_trajs_img = input_processing(sub_trajs_merc, self.model.lon_range, self.model.lat_range, 
                #                                 self.model.imgsize_x_lr, self.model.imgsize_y_lr,
                #                                 self.model.pixelrange_lr).to(self.device)

                # Meta Learning logic
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None: #and sub_trajs_idxs is not None:
                    # get the alpha weights of the current batch
                    alpha_batch = self.alpha[sub_trajs_idxs]  # shape=[B]
                    
                    # ============ use higher to do inner-loop update ============
                    with higher.innerloop_ctx(self.full_model, optimizer, track_higher_grads=False) as (meta_model, meta_opt):
                        # calculate the "reweighted" loss on the training batch
                        meta_embs = meta_model.encoder(sub_trajs_img)
                        meta_outs = meta_model.regression(meta_embs)
                        meta_pred_l1_simi = torch.cdist(meta_outs, meta_outs, 1)
                        
                        # only take the upper triangle
                        meta_mask_triu = torch.triu(torch.ones_like(meta_pred_l1_simi), diagonal=1).bool()
                        idx_i, idx_j = torch.where(meta_mask_triu)  # shape=[K], K = B*(B-1)/2

                        meta_pred_l1_simi = meta_pred_l1_simi[idx_i, idx_j]
                        meta_truth_l1_simi = sub_simi[idx_i, idx_j]
                        
                        # calculate the loss for each pair
                        loss_each_pair = (meta_pred_l1_simi - meta_truth_l1_simi)**2

                        # assign weights to each pair: w_ij = α_i + α_j
                        alpha_i = alpha_batch[idx_i]   # shape=[K]
                        alpha_j = alpha_batch[idx_j]   # shape=[K]
                        w_ij = alpha_i + alpha_j       # shape=[K]
                        
                        meta_loss = (w_ij * loss_each_pair).mean()
                        meta_opt.step(meta_loss)

                        # ============ calculate the loss of meta_model on the validation set, backpropagate to update alpha ============ 
                        val_trajs_img, sub_val_simi = self.sample_val_batch()

                        # use meta_model to forward on the validation set
                        val_outs = meta_model.encoder(val_trajs_img)
                        val_outs = meta_model.regression(val_outs)
                        
                        val_pred_l1_simi = torch.cdist(val_outs, val_outs, 1)
                        val_pred_l1_simi = val_pred_l1_simi[torch.triu(torch.ones(val_pred_l1_simi.shape), diagonal = 1) == 1]
                        val_truth_l1_simi = sub_val_simi[torch.triu(torch.ones(sub_val_simi.shape), diagonal = 1) == 1]

                        val_loss = self.criterion(val_pred_l1_simi, val_truth_l1_simi)
                    
                    # after with higher.innerloop_ctx(...), use the outer optimizer(here is opt_alpha) to update alpha
                    self.opt_alpha.zero_grad()
                    # gradient of val_loss w.r.t alpha (here the gradient of val_loss w.r.t alpha needs to be unrolled)
                    val_loss.backward()
                    self.opt_alpha.step()

                    # normal model training steps
                optimizer.zero_grad()
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None: # and sub_trajs_idxs is not None:
                    # use meta learning weights to train
                    self.model.zero_grad()
                    embs = self.full_model.encoder(sub_trajs_img)
                    outs = self.full_model.regression(embs)
                    pred_l1_simi = torch.cdist(outs, outs, 1)
                    mask_triu_2 = torch.triu(torch.ones_like(pred_l1_simi), diagonal=1) == 1
                    idx_i_2, idx_j_2 = torch.where(mask_triu_2)

                    dist_pred_ij_2 = pred_l1_simi[idx_i_2, idx_j_2]
                    dist_true_ij_2 = sub_simi[idx_i_2, idx_j_2]
                    loss_each_pair_2 = (dist_pred_ij_2 - dist_true_ij_2)**2

                    alpha_i_2 = self.alpha[sub_trajs_idxs][idx_i_2]
                    alpha_j_2 = self.alpha[sub_trajs_idxs][idx_j_2]
                    w_ij_2 = alpha_i_2 + alpha_j_2

                    train_loss = (w_ij_2 * loss_each_pair_2).mean()
                    train_loss.backward()
                    optimizer.step()
                else:

                    outs = self.full_model.regression(self.full_model.encoder(sub_trajs_img))
                    pred_l1_simi = torch.cdist(outs, outs, 1) # use l1 here.
                    pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                    truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                    train_loss = self.criterion(pred_l1_simi, truth_l1_simi)
                    train_loss.backward()
                    optimizer.step()

                train_losses.append(train_loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                # debug output
                if i_batch % 100 == 0 and Config.debug:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                .format(i_ep, i_batch, train_loss.item(), 
                                        time.time()-_time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))
                
                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                        and time.time() - training_starttime >= timetoreport[0]:
                    test_metrics = self.__test(self.dic_datasets['tests_img'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test.      ts={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()
                    self.regression.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_img'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            eval_hr_ep = np.mean(eval_metrics[1:])

            # early stopping
            if eval_hr_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = eval_hr_ep
                best_loss_train = tool_funcs.mean(train_losses)
                bad_counter = 0
                torch.save({'model': self.model.state_dict(), 'regression': self.regression.state_dict()}, 
                           self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == trjsr_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model) ))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.regression.load_state_dict(checkpoint['regression'])
        self.regression.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_img'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        test_endtime = time.time()
        logging.info("test. use time: {:.3f}s    loss= {:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(test_endtime - test_starttime, *test_metrics))
        

    def sample_val_batch(self):
        """
        sample a batch from the validation set, used for meta learning validation step
        """
        start_idx = random.randint(0, len(self.dic_datasets['evals_img']) - trjsr_batch_size - 1)
        trajs_imgs = self.dic_datasets['evals_img'][start_idx:start_idx+trjsr_batch_size]
        simi = self.dic_datasets['evals_simi'][start_idx:start_idx+trjsr_batch_size, start_idx:start_idx+trjsr_batch_size]
        max_distance = self.dic_datasets['max_distance']
        trajs_emb_cell, sub_simi = next(self._generate_standard_batch(simi, trajs_imgs, max_distance))
        return trajs_emb_cell, sub_simi
    
    # inner calling only
    @torch.no_grad()
    def __test(self, trajs_img, datasets_simi, max_distance):
        self.model.eval()
        self.regression.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_img)):
            sub_trajs_img = batch
            # sub_trajs_img = input_processing(sub_trajs_merc, self.model.lon_range, self.model.lat_range, 
            #                                     self.model.imgsize_x_lr, self.model.imgsize_y_lr,
            #                                     self.model.pixelrange_lr).to(self.device)
            outs = self.full_model.regression(self.full_model.encoder(sub_trajs_img))
            traj_embs.append(outs)

        traj_embs = torch.cat(traj_embs)
        pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)

        
        hr1 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 1, 1)
        hr5 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5)
        hrA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10)
        hrB = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50)
        hr5in1 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 1)
        hr10in5 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 5)
        hrBinA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10)


        return loss.item(), hr1, hr5, hrA, hrB, hr5in1, hr10in5, hrBinA
        

    def test(self):
        self.model.eval()
        self.regression.eval()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.regression.load_state_dict(checkpoint['regression'])
        self.regression.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_img'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        test_endtime = time.time()
        logging.info("test. use time: {:.3f}s    loss= {:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(test_endtime - test_starttime, *test_metrics))


    def trajsimi_dataset_generator_batchi(self, trajs_img):
        cur_index = 0
        len_datasets = len(trajs_img)
        
        while cur_index < len_datasets:
            end_index = cur_index + trjsr_batch_size \
                                if cur_index + trjsr_batch_size < len_datasets \
                                else len_datasets
            # sub_trajs_merc = [ [tool_funcs.meters2lonlat(p[0], p[1]) for p in trajs_merc[d_idx]] for d_idx in range(cur_index, end_index)]
            sub_trajs_img = trajs_img[cur_index: end_index].to(self.device)
            yield sub_trajs_img
            cur_index = end_index


    def trajsimi_dataset_generator_pairs_batchi(self, trajs_merc, max_distance, last_sampled_indices=None):
        if Config.strategy in ['mcts']:
            return self.trajsimi_dataset_generator_pairs_batchi_mcts(trajs_merc, max_distance)
        else:
            if Config.strategy in ['curriculum']:
                last_sampled_indices = None
            return self.trajsimi_dataset_generator_pairs_batchi_baseline(trajs_merc, max_distance, last_sampled_indices)
    
    
    def trajsimi_dataset_generator_pairs_batchi_baseline(self, train_traj, max_distance, last_sampled_indices=None, iteration_counts=iteration_counts):
        """
        sampled_indices: all trajectory data ids
        batch_traj_ids: trajectory data organized by batch
        batch_matrix_simis: similarity matrix organized by batch
        batch_traj_ids_top_k: topk trajectory data ids organized by batch
        """
        sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(self, train_traj, None, Config.trajsimi_measure, counts, \
                                                                                     trjsr_batch_size, strtegy=Config.strategy)
        
        if Config.strategy != 'curriculum':
            random.shuffle(sampled_indices)

        self.last_sampled_indices = sampled_indices #if Config.strategy != 'curriculum' else None

        # generate training batches
        # iteration = math.ceil(new_counts ** 2)
        batch_count = 0
        all_trajs = [train_traj[i] for i in sampled_indices]
        all_traj_imgs = self.trajs_merc_to_coor_to_img(all_trajs)
        traj_id_to_ori_id = {idx: i for i, idx in enumerate(sampled_indices)}

        while batch_count < iteration_counts * len(batch_traj_ids):
            # make sure to go through batch_count number of eval
            batch_idx = batch_count%len(batch_traj_ids)#random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]
            batch_traj_img = all_traj_imgs[[traj_id_to_ori_id[i] for i in batch_traj_id]].to(self.device)
            batch_train_sim = torch.tensor(batch_train_sim, device = self.device, dtype = torch.float) / max_distance
            batch_traj_id = torch.tensor(batch_traj_id, device=self.device, dtype=torch.long)
            yield batch_traj_img, batch_train_sim, batch_traj_id

            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # main loop meets this and does a test set evaluation


    def trajsimi_dataset_generator_pairs_batchi_mcts(self, train_traj, max_distance, iteration_counts=iteration_counts):
        # first use curriculum to warm up
        if self.mcts_pre_trained:
            sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(self, train_traj, None, Config.trajsimi_measure, counts, \
                                                                                     trjsr_batch_size, strtegy="curriculum")
        else:
            sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(
                self, train_traj, None, Config.trajsimi_measure, counts, trjsr_batch_size,
                strtegy=Config.strategy
            )

        batch_count = 0
        all_trajs = [train_traj[i] for i in sampled_indices]
        all_traj_imgs = self.trajs_merc_to_coor_to_img(all_trajs)
        traj_id_to_ori_id = {idx: i for i, idx in enumerate(sampled_indices)}

        while batch_count < iteration_counts * len(batch_traj_ids):
            # make sure to go through batch_count number of eval
            batch_idx = batch_count%len(batch_traj_ids)#random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]
            batch_traj_img = all_traj_imgs[[traj_id_to_ori_id[i] for i in batch_traj_id]].to(self.device)
            batch_train_sim = torch.tensor(batch_train_sim, device = self.device, dtype = torch.float) / max_distance
            batch_traj_id = torch.tensor(batch_traj_id, device=self.device, dtype=torch.long)
            yield batch_traj_img, batch_train_sim, batch_traj_id

            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # main loop meets this and does a test set evaluation

    def _generate_standard_batch(self, datasets_simi, trajs_img, max_distance):
        # len_datasets = len(trajs_img)
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        sub_trajs_img = trajs_img.to(self.device)
        yield sub_trajs_img, datasets_simi


    def load_trajsimi_dataset(self):
        # 1. read TrajSimi dataset
        # 2. convert merc_seq to cell_id_seqs
        
        dic_dataset = load_trajs(Config, train_split_ratio, eval_split_ratio, eval_num, test_num)
        eval_simis = dic_dataset['eval_simis']
        test_simis = dic_dataset['test_simis']
        max_distance = dic_dataset['max_distance']

        
        trains_merc = dic_dataset['train_trajs_merc']
        evals_merc = dic_dataset['eval_trajs_merc']
        tests_merc = dic_dataset['test_trajs_merc']
            
        evals_img = self.trajs_merc_to_coor_to_img(evals_merc)
        tests_img = self.trajs_merc_to_coor_to_img(tests_merc)

        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(trains_merc), len(evals_merc), len(tests_merc)))

        return {'trains_merc': trains_merc, 'evals_merc': evals_merc, 'tests_merc': tests_merc, \
                'evals_img': evals_img, 'tests_img': tests_img, \
                'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}


    def trajs_merc_to_coor_to_img(self, trajs_merc):
        lon_range = (Config.min_lon, Config.max_lon)
        lat_range = (Config.min_lat, Config.max_lat)

        trajs_coor = [ [tool_funcs.meters2lonlat(p[0], p[1]) for p in traj] for traj in trajs_merc]
        trajs_img_tensor = input_processing(trajs_coor, lon_range, lat_range, 
                                            Config.trjsr_imgsize_x_lr, Config.trjsr_imgsize_y_lr, 
                                            Config.trjsr_pixelrange_lr)
        return trajs_img_tensor

    def compute_mcts_reward(self, trajs_ids, trajs_simi):
        """
        calculate the test accuracy of the given trajectory set as MCTS reward
        """
        return self.compute_mcts_reward_legacy(trajs_ids, trajs_simi)

    def compute_mcts_reward_legacy(self, trajs_ids, trajs_simi):
        """
        the original reward function based on test accuracy
        """
        # 1. get the trajectory data
        trajs = [self.dic_datasets['trains_merc'][traj_id] for traj_id in trajs_ids]
        traj_imgs = self.trajs_merc_to_coor_to_img(trajs)
        max_distance = self.dic_datasets['max_distance']
        # 2. call __test to get hr
        _, hr1, hr5, hrA, hrB, hr5in1, hr10in5, hrBinA = self.__test(traj_imgs, trajs_simi, max_distance)
        # 3. calculate the reward
        return 1.0 - ((hr1 + hr5 + hrA + hr10in5 + hr5in1) / 5.0)
        


def parse_args():
    parser = argparse.ArgumentParser(description = "...")
    # dont give default value here! Otherwise, it will faultly overwrite the value in config.py.
    # config.py is the correct place to provide default values
    parser.add_argument('--debug', dest = 'debug', action='store_true')
    parser.add_argument('--dumpfile_uniqueid', type = str, help = '') # see config.py
    parser.add_argument('--seed', type = int, help = '')
    parser.add_argument('--dataset', type = str, help = '')
    parser.add_argument('--trajsimi_measure', type = str, help = '')
    
    parser.add_argument('--cell_embedding_dim', type = int, help = '')
    parser.add_argument('--traj_embedding_dim', type = int, help = '')
    parser.add_argument('--trajsimi_min_traj_len', type = int, help = '')
    parser.add_argument('--trajsimi_max_traj_len', type = int, help = '')
    parser.add_argument('--test', type = str, default="false", required=False, help = '')
    parser.add_argument('--gpu_id', type = int, help = '')
    # Meta Learning相关参数
    parser.add_argument('--use_meta_learning', action='store_true', 
                    help='use meta learning, use validation set to assist in correcting the training set weights')
    parser.add_argument('--cluster_num', type=int, default=10, 
                       help='number of clusters (default: 10)')
    parser.add_argument('--num_iterations', type=int, default=1000, 
                       help='number of MCTS iterations (default: 1000)')
    parser.add_argument('--budget', type=int, default=1000000, 
                       help='budget for MCTS (default: 1000000)')
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python TrjSR_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
if __name__ == '__main__':
    
    Config.strategy = 'mcts'
    Config.update(parse_args())
    Config.post_value_updates()
    logging.basicConfig(level = logging.DEBUG,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()]
            )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    tool_funcs.set_seed(Config.seed)

    trjsr = TrjSRTrainer()
    if Config.test is not None and Config.test.lower() == 'true':
        trjsr.test()
    else:
        trjsr.train()