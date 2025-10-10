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
import dgl
import numpy as np
import os

from config import Config as Config
from utilities import tool_funcs
from nn.TrajGAT_core import GraphTransformer, collate_fn, collate_fn_single
from nn.TrajGAT_utils import trajlist_to_trajgraph_parallel
from nn.utils.qtree import Index
from utilities.strategy_all import load_trajs, build_training_samples
import numpy as np
from nn.TrajGAT.utils.build_qtree import build_qtree
from nn.TrajGAT.utils.pre_embedding import get_pre_embedding


trajgat_epochs = 20
trajgat_batch_size = 64
trajgat_learning_rate = 0.001
trajgat_training_bad_patience = 5

train_split_ratio = 0.7
eval_split_ratio = 0.8
eval_num = 1000
test_num = 2000
counts = int(7000/trajgat_batch_size)
raw_inter_bad_patience = 10
iteration_counts = 200


class TrajGATTrainer:
    def __init__(self):
        super(TrajGATTrainer, self).__init__()
        self.device = torch.device('cuda:{}'.format(Config.gpu_id))
        torch.cuda.set_device(Config.gpu_id)
        # initialize the quadtree and coordinate range
        x_min, y_min = tool_funcs.lonlat2meters(Config.min_lon, Config.min_lat)
        x_max, y_max = tool_funcs.lonlat2meters(Config.max_lon, Config.max_lat)
        self.x_range = (x_min, x_max)
        self.y_range = (y_min, y_max)
        self.bounding_box = ((x_min, y_min), (x_max, y_max))
        
        logging.info("x_range: {}, y_range: {}".format(self.x_range, self.y_range))
        
        
        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_TrajGAT_{}_best{}_{}_{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid, Config.strategy, Config.use_meta_learning if hasattr(Config, 'use_meta_learning') else 'False')
        self.mcts = None
        self.last_sampled_indices = None

        # build the quadtree
        # if sys.path.
        # self.qtree = build_qtree(self.dic_datasets['trains_merc'], 
        #                             self.x_range, self.y_range, 
        #                             Config.trajgat_qtree_node_capacity, 50)
        max_depth = 10  
        qtree_file = '{}/{}_trajsimi_TrajGAT_qtree_{}_{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajgat_qtree_node_capacity, max_depth)
        if os.path.exists(qtree_file):
            with open(qtree_file, 'rb') as f:
                self.qtree, self.qtree_name2id, self.pre_embedding = pickle.load(f)
        else:
            self.qtree = build_qtree(
                self.dic_datasets['trains_merc'], 
                self.x_range, 
                self.y_range, 
                Config.trajgat_qtree_node_capacity, 
                max_depth
            )
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, Config.traj_embedding_dim)
            with open(qtree_file, 'wb') as f:
                pickle.dump((self.qtree, self.qtree_name2id, self.pre_embedding), f)
        
        logging.info("the quadtree is ok....")
            # decide whether to do embedding pre-training
        # self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, Config.traj_embedding_dim)
        
        # initialize the model
        self.model = GraphTransformer(
            d_input=2,  # (lon, lat, width, height)
            d_model=Config.traj_embedding_dim,
            num_head=Config.trajgat_num_head,
            num_encoder_layers=Config.trajgat_num_encoder_layers,
            d_lap_pos=Config.trajgat_d_lap_pos,
            encoder_dropout=Config.trajgat_encoder_dropout,
            pre_embedding=self.pre_embedding,  # 暂时不使用预训练embedding
            qtree=self.qtree,
            qtree_name2id=self.qtree_name2id,
            x_range=self.x_range,
            y_range=self.y_range
        )
        self.model.to(self.device)
        
        # Meta Learning
        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            N = len(self.dic_datasets['trains_merc'])  # 训练集总数
            # 定义一个可学习参数 alpha: shape=[N]
            alpha_init = getattr(Config, 'alpha_init', 1.0)
            alpha_init_tensor = torch.ones(N, dtype=torch.float, device=self.device) * alpha_init
            self.alpha = nn.Parameter(alpha_init_tensor)
            
            # alpha 的优化器(outer-most), 用于更新 alpha
            alpha_lr = getattr(Config, 'alpha_lr', 1e-4)
            self.opt_alpha = torch.optim.Adam([self.alpha], lr=alpha_lr)
            
        else:
            self.alpha = None
            self.opt_alpha = None

    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.set_anomaly_enabled(True)
    
        inter_bad_patience = raw_inter_bad_patience
        
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.model.parameters())), \
                                    lr = trajgat_learning_rate)
        
        # add a learning rate scheduler to help stable training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
        )

        best_hr_eval = -10000.0
        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            eval_metrics = self.__test(self.dic_datasets['evals_merc'], self.dic_datasets['evals_simi'], self.dic_datasets['max_distance'])
            logging.info("meta learning enabled")
            logging.info("initialize the best hr eval.  loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(*eval_metrics))
            best_hr_eval = np.mean(eval_metrics[1:])
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = trajgat_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change later

        for i_ep in range(trajgat_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []
            iter_bad_counter = 0
            inter_best_hr_eval = best_hr_eval

            self.model.train()
            # self.regression.train()

            for i_batch, batch in enumerate( self.trajsimi_dataset_generator_pairs_batchi( \
                                                            self.dic_datasets['trains_merc'], \
                                                            self.dic_datasets['max_distance'],\
                                                         last_sampled_indices=self.last_sampled_indices)):
                
                if batch == 'EVAL':
                    eval_metrics = self.__test(self.dic_datasets['evals_merc'], self.dic_datasets['evals_simi'], self.dic_datasets['max_distance'])
                    logging.info("eval.     step={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(i_batch, *eval_metrics))
                    eval_hr_ep = np.mean(eval_metrics[1:])
                    # eval_loss_ep = eval_metrics[0]
                    if eval_hr_ep > inter_best_hr_eval:
                        iter_bad_counter = 0
                        inter_best_hr_eval = eval_hr_ep
                        torch.save({'model': self.model.state_dict()}, 
                           self.checkpoint_path)
                        continue
                    else:
                        iter_bad_counter += 1
                        if iter_bad_counter == inter_bad_patience:
                            logging.info("No improvement at step {}, early stopping.".format(i_batch))
                            # load the best model
                            self.model.load_checkpoint(self.checkpoint_path, self.device)
                            break
                        else:
                            continue
                _time_batch = time.time()

                sub_trajs_merc, sub_simi, sub_trajs_idxs = batch

                # Meta Learning logic
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None:
                    # get the alpha weight of the current batch
                    alpha_batch = self.alpha[sub_trajs_idxs]  # shape=[B]
                    
                    # ============ use higher for inner-layer update ============
                    with higher.innerloop_ctx(self.model, optimizer, track_higher_grads=False) as (meta_model, meta_opt):
                        # calculate the "reweighted" loss on the training batch
                        meta_outs = meta_model(sub_trajs_merc)
                        # meta_pred_l1_simi = torch.square(torch.cdist(meta_outs, meta_outs, 2))
                        meta_pred_l1_simi = torch.cdist(meta_outs, meta_outs, 1)
                        
                        # only take the upper triangle
                        meta_mask_triu = torch.triu(torch.ones_like(meta_pred_l1_simi), diagonal=1).bool()
                        idx_i, idx_j = torch.where(meta_mask_triu)  # each is shape=[K], K = B*(B-1)/2

                        meta_pred_l1_simi = meta_pred_l1_simi[idx_i, idx_j]
                        meta_truth_l1_simi = sub_simi[idx_i, idx_j]
                        
                        # calculate the loss of each pair
                        loss_each_pair = (meta_pred_l1_simi - meta_truth_l1_simi)**2

                        # assign the weight factor to each pair: w_ij = α_i + α_j
                        alpha_i = alpha_batch[idx_i]   # shape=[K]
                        alpha_j = alpha_batch[idx_j]   # shape=[K]
                        w_ij = alpha_i + alpha_j       # shape=[K]
                        
                        meta_loss = (w_ij * loss_each_pair).mean()
                        meta_opt.step(meta_loss)

                        # ============ calculate the loss of meta_model on the validation set, backpropagate to update alpha ============ 
                        val_trajs_merc, sub_val_simi = self.sample_val_batch()

                        # use meta_model to forward on the validation set
                        val_embs = meta_model(val_trajs_merc)
                        # val_embs = torch.nn.functional.normalize(val_embs, p=2, dim=-1)
                        val_outs = val_embs
                        # val_pred_l1_simi = torch.square(torch.cdist(val_outs, val_outs, 2))
                        val_pred_l1_simi = torch.cdist(val_outs, val_outs, 1)
                        # val_pred_l1_simi = 1 - torch.mm(val_outs, val_outs.T)
                        val_pred_l1_simi = val_pred_l1_simi[torch.triu(torch.ones(val_pred_l1_simi.shape), diagonal = 1) == 1]
                        val_truth_l1_simi = sub_val_simi[torch.triu(torch.ones(sub_val_simi.shape), diagonal = 1) == 1]

                        val_loss = self.criterion(val_pred_l1_simi, val_truth_l1_simi)
                    
                    
                    # after with higher.innerloop_ctx(...), use the outer optimizer(here is opt_alpha) to update alpha
                    if self.opt_alpha is not None:
                        self.opt_alpha.zero_grad()
                        # val_loss w.r.t alpha (here val_loss w.r.t alpha needs to be unrolled)
                        val_loss.backward()
                        self.opt_alpha.step()

                # normal model training step
                optimizer.zero_grad()
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None:
                    # use meta learning weights to train
                    self.model.zero_grad()
                    embs = self.model(sub_trajs_merc)
                    outs = embs
                    # pred_l1_simi = torch.square(torch.cdist(outs, outs, 2))
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
                    # add gradient clipping to prevent gradient explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    try:
                        # check if the input data has NaN
                        if torch.isnan(sub_trajs_merc.ndata["feat"]).any():
                            logging.warning("Input features contain NaN, skipping this batch")
                            optimizer.zero_grad()
                            continue
                            
                        outs = self.model(sub_trajs_merc)
                        
                        # check if the model output has NaN
                        if torch.isnan(outs).any():
                            logging.warning("Model output contains NaN, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        # pred_l1_simi = torch.square(torch.cdist(outs, outs, 2)) # use l1 here.
                        pred_l1_simi = torch.cdist(outs, outs, 1)
                        
                        # check if the distance calculation has NaN
                        if torch.isnan(pred_l1_simi).any():
                            logging.warning("Distance computation contains NaN, skipping this batch")
                            optimizer.zero_grad()
                            continue
                            
                        pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                        truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                        train_loss = self.criterion(pred_l1_simi, truth_l1_simi)
                        
                        # check if the loss value is NaN
                        if torch.isnan(train_loss) or torch.isinf(train_loss):
                            logging.warning(f"NaN or Inf loss detected: {train_loss.item()}, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        # check if the loss value is too large
                        if train_loss.item() > 1000000:
                            logging.warning(f"Loss too large: {train_loss.item()}, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        train_loss.backward()
                        
                        # check if the gradient has NaN
                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            logging.warning("Gradients contain NaN, skipping this batch")
                            optimizer.zero_grad()
                            continue
                        
                        # add gradient clipping to prevent gradient explosion
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                    except Exception as e:
                        logging.error(f"Error in training batch: {e}")
                        optimizer.zero_grad()
                        continue

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
                    test_metrics = self.__test(self.dic_datasets['tests_merc'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test.      ts={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()
                    # self.regression.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_merc'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
            logging.info("eval.     i_ep={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(i_ep, *eval_metrics))
            eval_hr_ep = np.mean(eval_metrics[1:])
            
            # update the learning rate scheduler
            scheduler.step(eval_metrics[0])  # use the validation loss to adjust the learning rate

            # early stopping
            if eval_hr_ep > best_hr_eval:
                best_epoch = i_ep
                best_hr_eval = eval_hr_ep
                best_loss_train = tool_funcs.mean(train_losses)
                bad_counter = 0
                torch.save({'model': self.model.state_dict()}, 
                           self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == trajgat_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model) ))
                break
            
        # test
        self.model.load_checkpoint(self.checkpoint_path, self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_merc'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'])
        test_endtime = time.time()
        logging.info("test. use time: {:.3f}s    loss= {:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(test_endtime - test_starttime, *test_metrics))
        

    def sample_val_batch(self):
        """
        sample a batch from the validation set, for the validation step of meta learning
        """
        start_idx = random.randint(0, len(self.dic_datasets['evals_merc']) - trajgat_batch_size-1)
        trajs_merc = self.dic_datasets['evals_merc'][start_idx:start_idx+trajgat_batch_size]
        traj_graphs = self._prepare_batch_graphs(trajs_merc)
        simi = self.dic_datasets['evals_simi'][start_idx:start_idx+trajgat_batch_size, start_idx:start_idx+trajgat_batch_size]
        max_distance = self.dic_datasets['max_distance']
        trajs_merc, sub_simi, _ = next(self._generate_standard_batch(list(range(len(trajs_merc))), simi, traj_graphs, max_distance))
        return trajs_merc, sub_simi
    
    def _prepare_batch_graphs(self, trajs_merc):
        """
        convert the trajectory list to DGL graph batch
        """
        lon_range = (Config.min_lon, Config.max_lon)
        lat_range = (Config.min_lat, Config.max_lat)
        
        # convert the trajectory to graph
        # logging.info(f"Preparing batch graphs, trajs_merc: {len(trajs_merc)}")
        traj_graphs = trajlist_to_trajgraph_parallel(trajs_merc, self.qtree, self.qtree_name2id, lon_range, lat_range)
        # logging.info(f"end prepare batch graphs, length: {len(traj_graphs)}")
        return traj_graphs
        
        # # batch the graphs
        # batched_graph = dgl.batch(traj_graphs)
        # return batched_graph.to(self.device)
    
    def test(self):
        self.model.eval()
        self.__test(self.dic_datasets['tests_merc'], \
                    self.dic_datasets['tests_simi'], \
                    self.dic_datasets['max_distance'])
        
    # inner calling only
    @torch.no_grad()
    def __test(self, trajs_merc, datasets_simi, max_distance):
        self.model.eval()
        # self.regression.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance

        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_merc)):
            sub_trajs_merc = batch
            traj_graphs = self._prepare_batch_graphs(sub_trajs_merc)
            traj_graphs = dgl.batch(traj_graphs).to(self.device)
            # # need to normalize the feature in the graph
            with torch.no_grad():
                feats = traj_graphs.ndata['feat'][:, :2]
                traj_graphs.ndata['feat'] = feats
                xs = feats[:, 0]
                ys = feats[:, 1]
                meanx, meany = xs.mean(), ys.mean()
                stdx, stdy = xs.std(), ys.std()
                feats[:, 0] = (xs - meanx) / (stdx + 1e-8)
                feats[:, 1] = (ys - meany) / (stdy + 1e-8)
                traj_graphs.ndata['feat'] = feats
            embs = self.model(traj_graphs)
            outs = embs
            traj_embs.append(outs)

        traj_embs = torch.cat(traj_embs)
        # pred_l1_simi = torch.square(torch.cdist(traj_embs, traj_embs, 2))
        pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)

        
        hr1 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 1, 1, fixed=True)
        hr5 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 5, fixed=True)
        hrA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 10, fixed=True)
        hrB = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 50, fixed=True)
        hr5in1 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 5, 1, fixed=True)
        hr10in5 = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 10, 5, fixed=True)
        hrBinA = tool_funcs.hitting_ratio(pred_l1_simi, truth_l1_simi, 50, 10, fixed=True)


        return loss.item(), hr1, hr5, hrA, hrB, hr5in1, hr10in5, hrBinA
        

    def trajsimi_dataset_generator_batchi(self, trajs_merc):
        cur_index = 0
        len_datasets = len(trajs_merc)
        
        while cur_index < len_datasets:
            end_index = cur_index + trajgat_batch_size \
                                if cur_index + trajgat_batch_size < len_datasets \
                                else len_datasets
            sub_trajs_merc = trajs_merc[cur_index: end_index]
            # need to normalize the feature in the graph
            yield sub_trajs_merc
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
                                                                                     trajgat_batch_size, strtegy=Config.strategy)
        
        if Config.strategy != 'curriculum':
            random.shuffle(sampled_indices)

        self.last_sampled_indices = sampled_indices #if Config.strategy != 'curriculum' else None

        # generate training batches
        batch_count = 0
        traj_graphs_list = self._prepare_batch_graphs([train_traj[i] for i in sampled_indices])
        traj_id_to_original_idx = {i: idx for idx, i in enumerate(sampled_indices)}
        while batch_count < iteration_counts * len(batch_traj_ids):
            batch_idx = batch_count%len(batch_traj_ids)#random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]

            batch_traj_merc = [traj_graphs_list[traj_id_to_original_idx[i]] for i in batch_traj_id]
            yield from self._generate_standard_batch(
                    batch_traj_id, batch_train_sim, batch_traj_merc, max_distance
                )
            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # main loop meets this to do one evaluation on the test set


    def trajsimi_dataset_generator_pairs_batchi_mcts(self, train_traj, max_distance, iteration_counts=iteration_counts):
        sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(
            self, train_traj, None, Config.trajsimi_measure, counts, trajgat_batch_size,
            strtegy=Config.strategy
        )

        batch_count = 0
        traj_graphs_list = self._prepare_batch_graphs([train_traj[i] for i in sampled_indices])
        traj_id_to_original_idx = {i: idx for idx, i in enumerate(sampled_indices)}
        
        while batch_count < iteration_counts * len(batch_traj_ids):
            # make sure to do one evaluation after batch_count number of batches
            batch_idx = batch_count%len(batch_traj_ids)#random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]
            batch_traj_merc = [traj_graphs_list[traj_id_to_original_idx[i]] for i in batch_traj_id]
            
            yield from self._generate_standard_batch(
                batch_traj_id, batch_train_sim, batch_traj_merc, max_distance
            )
            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # main loop meets this to do one evaluation on the test set

    def _generate_standard_batch(self, batch_traj_id, datasets_simi, trajs_merc, max_distance):
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        sub_trajs_idxs = torch.tensor(batch_traj_id, device=self.device, dtype=torch.long)
        traj_graphs = dgl.batch(trajs_merc).to(self.device)
        with torch.no_grad():
            feats = traj_graphs.ndata['feat'][:, :2]
            traj_graphs.ndata['feat'] = feats
            xs = feats[:, 0]
            ys = feats[:, 1]
            meanx, meany = xs.mean(), ys.mean()
            stdx, stdy = xs.std(), ys.std()
            feats[:, 0] = (xs - meanx) / (stdx + 1e-8)
            feats[:, 1] = (ys - meany) / (stdy + 1e-8)
            traj_graphs.ndata['feat'] = feats
        yield traj_graphs, datasets_simi, sub_trajs_idxs


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

        logging.info("trajsimi dataset sizes. max_distance={}, (trains/evals/tests={}/{}/{})" \
                    .format(max_distance, len(trains_merc), len(evals_merc), len(tests_merc)))

        return {'trains_merc': trains_merc, 'evals_merc': evals_merc, 'tests_merc': tests_merc, \
                'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance, 'eval_traj': evals_merc}


    def compute_mcts_reward(self, trajs_ids, trajs_simi):
        """
        calculate the test accuracy of the given trajectory set as MCTS reward
        """
        # compatible with legacy reward
        if hasattr(Config, 'use_legacy_reward') and Config.use_legacy_reward:
            return self.compute_mcts_reward_legacy(trajs_ids, trajs_simi)
        # default to use legacy
        return self.compute_mcts_reward_legacy(trajs_ids, trajs_simi)

    def compute_mcts_reward_legacy(self, trajs_ids, trajs_simi):
        """
        the original reward function based on test accuracy
        """
        # 1. get the trajectory data
        # 2. sample
        trajs_ids = trajs_ids[:10]
        trajs_simi = trajs_simi[:10, :10]
        trajs = [self.dic_datasets['trains_merc'][traj_id] for traj_id in trajs_ids]
        max_distance = self.dic_datasets['max_distance']

        # 3. calculate the loss
        self.model.eval()
        # self.regression.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(trajs_simi, device = self.device, dtype = torch.float) / max_distance

        traj_graphs = self._prepare_batch_graphs(trajs)
        traj_graphs = dgl.batch(traj_graphs).to(self.device)
        # # need to normalize the feature in the graph
        with torch.no_grad():
            feats = traj_graphs.ndata['feat'][:, :2]
            traj_graphs.ndata['feat'] = feats
            embs = self.model(traj_graphs)
            outs = embs
            traj_embs.append(outs)

        traj_embs = torch.cat(traj_embs)
        # pred_l1_simi = torch.square(torch.cdist(traj_embs, traj_embs, 2))
        pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
        truth_l1_simi = datasets_simi
        pred_l1_simi_seq = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
        truth_l1_simi_seq = truth_l1_simi[torch.triu(torch.ones(truth_l1_simi.shape), diagonal = 1) == 1]

        loss = self.criterion(pred_l1_simi_seq, truth_l1_simi_seq)

        return loss.item()

        # # 2. call __test to get hr
        # _, hr1, hr5, hrA, hrB, hr5in1, hr10in5, hrBinA = self.__test(trajs, trajs_simi, max_distance)
        # # 3. calculate the reward
        # return 1.0 - ((hr1 + hr5 + hrA + hr10in5 + hr5in1) / 5.0)


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
    
    # Meta Learning related parameters
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


# nohup python TrajGAT_train_script_all.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
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

    trajgat = TrajGATTrainer()
    if Config.test is not None and Config.test.lower() == 'true':
        trajgat.test()
    else:
        trajgat.train()