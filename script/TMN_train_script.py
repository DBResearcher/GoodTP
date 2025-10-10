import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
import logging
import argparse
import random
import higher
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config as Config
from utilities import tool_funcs
from nn.utils.cellspace import CellSpace
from nn.TMN import TMN
from nn.NEUTRAJ_utils import distance_sampling, negative_distance_sampling, pad_sequence
from utilities.strategy_all import load_trajs, build_training_samples


tmn_grid_delta = 100
tmn_epochs = 10
tmn_batch_size = 256
tmn_learning_rate = 0.001
tmn_training_bad_patience = 5
tmn_batch_size_testing = 256
counts = int(7000/tmn_batch_size_testing)

train_split_ratio = 0.7
eval_split_ratio = 0.8
eval_num = 1000
test_num = 2000
iteration_counts = 200
# inter_bad_patience = 5
inter_bad_patience = 20


class TMNTrainer:
    def __init__(self):
        super(TMNTrainer, self).__init__()
        self.device = torch.device('cuda:{}'.format(Config.gpu_id))
        x_min, y_min = tool_funcs.lonlat2meters(Config.min_lon, Config.min_lat)
        x_max, y_max = tool_funcs.lonlat2meters(Config.max_lon, Config.max_lat)
        self.cellspace = CellSpace(Config.cell_size, Config.cell_size, 
                                    x_min, y_min, x_max, y_max)
        self.bounding_box = ((x_min, y_min), (x_max, y_max))

        self.model = TMN(4, Config.traj_embedding_dim, (self.cellspace.x_size, self.cellspace.y_size), 
                    Config.tmn_sampling_num, (x_min, x_max), (y_min, y_max), Config.cell_size)
        self.model.to(self.device)
        
        self.dic_datasets = self.load_trajsimi_dataset()
        
        self.checkpoint_path = '{}/{}_trajsimi_TMN_{}_best{}_{}_{}.pt'.format(Config.snapshot_dir, \
                                    Config.dataset_prefix, Config.trajsimi_measure, Config.dumpfile_uniqueid, Config.strategy, (Config.use_meta_learning if hasattr(Config, 'use_meta_learning') else 'False'))
        self.mcts = None
        self.last_sampled_indices = None
        # Meta Learning
        if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning:
            N = len(self.dic_datasets['train_traj'])
            alpha_init = getattr(Config, 'alpha_init', 1.0)
            alpha_init_tensor = torch.ones(N, dtype=torch.float, device=self.device) * alpha_init
            self.alpha = nn.Parameter(alpha_init_tensor)
            alpha_lr = getattr(Config, 'alpha_lr', 1e-4)
            self.opt_alpha = torch.optim.Adam([self.alpha], lr=alpha_lr)
            # load pre-trained model
            logging.info("self.checkpoint_path: {}".format(self.checkpoint_path))
            self.model.load_state_dict(torch.load(self.checkpoint_path.replace('_True', '_False'))['model'])
        else:
            self.alpha = None
            self.opt_alpha = None

    def train(self):
        logging.info("training. START! @={:.3f}".format(time.time()))
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        torch.autograd.set_detect_anomaly(True)
        
        # self.criterion = WeightedRankingLoss(batch_size = tmn_batch_size, sampling_num = Config.tmn_sampling_num)
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
                                            lr = tmn_learning_rate)

        best_hr_eval = -10000000.0
        best_loss_train = 10000000.0
        best_epoch = 0
        bad_counter = 0
        bad_patience = tmn_training_bad_patience
        timetoreport = [1200, 2400, 3600] # len may change late
        cosume_budget = 0
        for i_ep in range(tmn_epochs):
            _time_ep = time.time()
            train_losses = []
            train_gpu = []
            train_ram = []
            iter_bad_counter = 0
            inter_best_hr_eval = best_hr_eval

            if cosume_budget > Config.budget:
                logging.info("budget exceeded, early stopping.")
                break
            for i_batch, batch in enumerate(self.trajsimi_dataset_generator_pairs_batchi(self.dic_datasets['train_traj'], self.dic_datasets['max_distance'], last_sampled_indices=self.last_sampled_indices)):
                if batch == 'EVAL':
                    eval_metrics = self.__test(self.dic_datasets['evals_trajcoorgrid'], \
                                        self.dic_datasets['evals_simi'], \
                                        self.dic_datasets['max_distance'])
                    logging.info("eval.     step={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(i_batch, *eval_metrics))
                    eval_hr_ep = np.mean(eval_metrics[1:])
                    if eval_hr_ep > inter_best_hr_eval:
                        iter_bad_counter = 0
                        inter_best_hr_eval = eval_hr_ep
                        torch.save({'model': self.model.state_dict()}, self.checkpoint_path)
                        continue
                    else:
                        iter_bad_counter += 1
                        if iter_bad_counter == inter_bad_patience:
                            logging.info("No improvement at step {}, early stopping.".format(i_batch))
                            checkpoint = torch.load(self.checkpoint_path)
                            self.model.load_state_dict(checkpoint['model'])
                            self.model.to(self.device)
                            break
                        else:
                            continue

                self.model.train()
                _time_batch = time.time()
                
                # get batch data - TMN format
                inputs_arrays, inputs_len_arrays, sub_simi = batch[0], batch[1], batch[2]
                inputs0 = torch.Tensor(inputs_arrays).to(self.device)
                inputs1 = torch.Tensor(inputs_arrays).to(self.device)
                batch_traj_ids = batch[3] if len(batch) > 3 else None
                # logging.info("inputs: {}, {}, traj_ids:{}, sub_simi:{}".format(inputs0.shape, inputs0[:10], batch_traj_ids, sub_simi[:10, :10]))
                # sys.exit()

                # Meta Learning logic - TMN format
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None:
                    # get the alpha weights of the current batch
                    if batch_traj_ids is not None:
                        alpha_batch = self.alpha[batch_traj_ids]  # shape=[B]
                        
                        # ============ use higher for inner-layer update ============
                        with higher.innerloop_ctx(self.model, optimizer, track_higher_grads=False) as (meta_model, meta_opt):
                            # calculate the "reweighted" loss on the training batch
                            meta_outs = meta_model.smn.f_single(inputs0, inputs_len_arrays)
                            # meta_outs = (embs0+embs1)/2.0
                            meta_pred_l1_simi = 1 - torch.nn.functional.cosine_similarity(meta_outs.unsqueeze(1), meta_outs.unsqueeze(0), dim=-1) #torch.cdist(meta_outs, meta_outs, p=1) #torch.exp(torch.cdist(meta_outs, meta_outs, p=2))
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
                            sub_val_inputs, sub_val_inputs_len, sub_val_simi = self.sample_val_batch()
                            val_inputs0 = torch.Tensor(sub_val_inputs).to(self.device)
                            
                            # use meta_model to forward on the validation set
                            val_outs = meta_model.smn.f_single(val_inputs0, sub_val_inputs_len)
                            # val_outs = (val_emb0+val_emb1 )/2.0
                            val_pred_l1_simi = 1 - torch.nn.functional.cosine_similarity(val_outs.unsqueeze(1), val_outs.unsqueeze(0), dim=-1) #torch.cdist(val_outs, val_outs, 1) #torch.exp(torch.cdist(val_outs, val_outs, 2))
                            val_pred_l1_simi = val_pred_l1_simi[torch.triu(torch.ones(val_pred_l1_simi.shape), diagonal = 1) == 1]
                            val_truth_l1_simi = sub_val_simi[torch.triu(torch.ones(sub_val_simi.shape), diagonal = 1) == 1]

                            val_loss = self.criterion(val_pred_l1_simi, val_truth_l1_simi)
                        # after with higher.innerloop_ctx(...), use the outer optimizer(here is opt_alpha) to update alpha
                        self.opt_alpha.zero_grad()
                        # the gradient of val_loss w.r.t alpha(here the gradient of val_loss w.r.t alpha needs to be unrolled)
                        val_loss.backward()
                        self.opt_alpha.step()

                # normal model training steps - TMN format
                optimizer.zero_grad()
                if hasattr(Config, 'use_meta_learning') and Config.use_meta_learning and self.alpha is not None and batch_traj_ids is not None:
                    # use meta learning weights to train
                    self.model.zero_grad()
                    outs = self.model.smn.f_single(inputs0, inputs_len_arrays)
                    # outs = (embs0+embs1)/2.0
                            
                    pred_l1_simi = 1 - torch.nn.functional.cosine_similarity(outs.unsqueeze(1), outs.unsqueeze(0), dim=-1) #torch.cdist(outs, outs, 1) #torch.exp(torch.cdist(outs, outs, 2))
                    mask_triu_2 = torch.triu(torch.ones_like(pred_l1_simi), diagonal=1) == 1
                    idx_i_2, idx_j_2 = torch.where(mask_triu_2)

                    dist_pred_ij_2 = pred_l1_simi[idx_i_2, idx_j_2]
                    dist_true_ij_2 = sub_simi[idx_i_2, idx_j_2]
                    loss_each_pair_2 = (dist_pred_ij_2 - dist_true_ij_2)**2

                    alpha_i_2 = self.alpha[batch_traj_ids][idx_i_2]
                    alpha_j_2 = self.alpha[batch_traj_ids][idx_j_2]
                    w_ij_2 = alpha_i_2 + alpha_j_2

                    train_loss = (w_ij_2 * loss_each_pair_2).mean()
                    train_loss.backward()
                    optimizer.step()
                else:
                    # standard training process
                    outs = self.model.smn.f_single(inputs0, inputs_len_arrays)
                    
                    # outs = (embs0+embs1)/2.0
                    pred_l1_simi = 1 - torch.nn.functional.cosine_similarity(outs.unsqueeze(1), outs.unsqueeze(0), dim=-1) #torch.cdist(outs, outs, 1) #torch.exp(torch.cdist(outs, outs, 2))
                    # logging.info("outs: {}, {}, pred_l1_simi:{}".format(outs.shape, outs[:10], pred_l1_simi[:10, :10]))
                    # sys.exit()
                    pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal = 1) == 1]
                    truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal = 1) == 1]
                    train_loss = self.criterion(pred_l1_simi, truth_l1_simi)
                    train_loss.backward()
                    optimizer.step()

                train_losses.append(train_loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())
                cosume_budget += len(sub_simi)
                # debug output, only for debug  
                if i_batch % 100 == 0 and Config.debug:
                    logging.debug("training. ep-batch={}-{}, train_loss={:.4f}, @={:.3f}, gpu={}, ram={}" \
                                .format(i_ep, i_batch, train_loss.item(), 
                                        time.time()-_time_batch, tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

                # exp of training time vs. effectiveness
                if Config.trajsimi_timereport_exp and len(timetoreport) \
                        and time.time() - training_starttime >= timetoreport[0]:
                    test_metrics = self.__test(self.dic_datasets['tests_trajcoorgrid'], \
                                                self.dic_datasets['tests_simi'], \
                                                self.dic_datasets['max_distance'])
                    logging.info("test.     ts={}, loss={:.4f}, hr={:.3f},{:.3f},{:.3f}".format(timetoreport[0], *test_metrics))
                    timetoreport.pop(0)
                    self.model.train()

            # ep debug output
            logging.info("training. i_ep={}, loss={:.4f}, @={:.3f}" \
                        .format(i_ep, tool_funcs.mean(train_losses), time.time()-_time_ep))
            
            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # eval
            eval_metrics = self.__test(self.dic_datasets['evals_trajcoorgrid'], \
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
                torch.save({'model': self.model.state_dict()}, self.checkpoint_path)
            else:
                bad_counter += 1

            if bad_counter == bad_patience or i_ep + 1 == tmn_epochs:
                training_endtime = time.time()
                logging.info("training end. @={:.0f}, best_epoch={}, best_loss_train={:.4f}, best_hr_eval={:.4f}, #param={}" \
                            .format(training_endtime - training_starttime, \
                                    best_epoch, best_loss_train, best_hr_eval, \
                                    tool_funcs.num_of_model_params(self.model) ))
                break
            
        # test
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_trajcoorgrid'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'], False)
        test_endtime = time.time()
        logging.info("test.     loss= {:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(*test_metrics))
        return {'task_train_time': training_endtime - training_starttime, \
                'task_train_gpu': training_gpu_usage, \
                'task_train_ram': training_ram_usage, \
                'task_test_time': test_endtime - test_starttime, \
                'task_test_gpu': 0, \
                'task_test_ram': 0, \
                'hr10': test_metrics[1], 'hr50': test_metrics[2], 'hr50in10': test_metrics[3]}

    def test(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        test_starttime = time.time()
        test_metrics = self.__test(self.dic_datasets['tests_trajcoorgrid'], \
                                    self.dic_datasets['tests_simi'], \
                                    self.dic_datasets['max_distance'], False)
        test_endtime = time.time()
        logging.info("test.  use time:{:.0f},   loss= {:.4f}, hr={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\
                     ".format(test_endtime - test_starttime, *test_metrics))

    # inner calling only
    @torch.no_grad()
    def __test(self, trajs_coorgrid, datasets_simi, max_distance, shuffle=True):
        self.model.eval()
        
        traj_embs = []
        datasets_simi = torch.tensor(datasets_simi, device = self.device, dtype = torch.float) / max_distance
        
        for i_batch, batch in enumerate(self.trajsimi_dataset_generator_batchi(trajs_coorgrid)):
            inputs_arrays, inputs_len_arrays = batch
            inputs0 = torch.Tensor(inputs_arrays[0]).to(self.device)
            inputs1 = torch.Tensor(inputs_arrays[1]).to(self.device)
            outs = self.model.smn.f_single(inputs0, inputs_len_arrays[0])
            # outs = (embs0+embs1)/2.0
            traj_embs.append(outs)
        traj_embs = torch.cat(traj_embs)

        pred_l1_simi = 1 - torch.nn.functional.cosine_similarity(traj_embs.unsqueeze(1), traj_embs.unsqueeze(0), dim=-1)
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

    def compute_mcts_reward(self, trajs_ids, trajs_simi):
        # check if use legacy mode
        return self.compute_mcts_reward_legacy(trajs_ids, trajs_simi)
        
    def compute_mcts_reward_legacy(self, trajs_ids, trajs_simi):
        """
        the original reward function based on test accuracy (kept as a backup)
        """
        # given the id of the trajectory data, get the corresponding trajectory data from the training set, then call the test function to get the hr value, and finally average the reward
        # since the calculation is too slow, we randomly sample 60 for testing
        # sampled_indices = random.sample(range(len(trajs_ids)), min(60, len(trajs_ids)))
        # # 1. get the trajectory data
        # trajs_ids = [trajs_ids[idx] for idx in sampled_indices]
        # trajs_simi = trajs_simi[sampled_indices, :][:, sampled_indices]
        trajs = [self.dic_datasets['train_traj'][traj_id] for traj_id in trajs_ids]
        trajs_coorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(trajs)
        max_distance = self.dic_datasets['max_distance']

        # 2. call the test function to get the hr value
        _, hr1, hr5, hrA, hrB, hr5in1, hr10in5, hrBinA = self.__test(trajs_coorgrid, \
                                                                     trajs_simi, max_distance)
        # 3. calculate the reward: the better the result, the smaller the reward
        return 1.0 - ((hr1 + hr5 + +hrA + hr10in5+ hr5in1) / 5.0)

    def sample_val_batch(self):
        """
        sample a batch from the validation set, for the validation step of meta learning
        """
        start_idx = random.randint(0, len(self.dic_datasets['evals_trajcoorgrid']) - tmn_batch_size_testing-1)
        trajs_coorgrid = self.dic_datasets['evals_trajcoorgrid'][start_idx:start_idx+tmn_batch_size_testing]
        simi = self.dic_datasets['evals_simi'][start_idx:start_idx+tmn_batch_size_testing, start_idx:start_idx+tmn_batch_size_testing]
        max_distance = self.dic_datasets['max_distance']
        traj_id_to_ori_id = {idx: i for i, idx in enumerate(range(len(trajs_coorgrid)))}
        inputs_arrays, inputs_len_arrays, distance_arrays, _ = next(self._generate_standard_batch(list(range(len(trajs_coorgrid))), simi, trajs_coorgrid, traj_id_to_ori_id, max_distance))
      
        return inputs_arrays, inputs_len_arrays, distance_arrays

    
    def trajsimi_dataset_generator_batchi(self, trajs_coorgrid):
        cur_index = 0
        len_datasets = len(trajs_coorgrid)
        trajs_len = [len(traj) for traj in trajs_coorgrid]
        while cur_index < len_datasets:
            end_index = cur_index + tmn_batch_size_testing \
                                if cur_index + tmn_batch_size_testing < len_datasets \
                                else len_datasets
            anchor_input, trajs_input = [], []
            anchor_input_len, trajs_input_len = [], []
            for d_idx in range(cur_index, end_index):
                anchor_input.append(trajs_coorgrid[d_idx])
                trajs_input.append(trajs_coorgrid[d_idx])
                anchor_input_len.append(trajs_len[d_idx])
                trajs_input_len.append(trajs_len[d_idx])
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            yield ([np.array(anchor_input),np.array(trajs_input)],
                   [anchor_input_len, trajs_input_len])
            cur_index = end_index
        

    def trajsimi_dataset_generator_pairs_batchi(self, trajs_merc, max_distance, last_sampled_indices=None):
        if Config.strategy in ['mcts']:
            return self.trajsimi_dataset_generator_pairs_batchi_mcts(trajs_merc, max_distance)
        else:
            if Config.strategy in ['curriculum']:
                last_sampled_indices = None
            return self.trajsimi_dataset_generator_pairs_batchi_baseline(trajs_merc, max_distance, last_sampled_indices)

    def trajsimi_dataset_generator_pairs_batchi_mcts(self, trajs_merc, max_distance, iteration_counts=iteration_counts):
        sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(
            self, trajs_merc, None, Config.trajsimi_measure, counts, tmn_batch_size_testing,
            strtegy=Config.strategy
        )

        sampled_train_traj = [trajs_merc[idx] for idx in sampled_indices]
        sampled_train_trajs_coorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(sampled_train_traj)
        traj_id_to_ori_id = {idx: i for i, idx in enumerate(sampled_indices)}

        batch_count = 0
        while batch_count < iteration_counts * len(batch_traj_ids):
            # make sure to go through batch_count number of eval
            batch_idx = random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]
            yield from self._generate_standard_batch(
                batch_traj_id, batch_train_sim,
                sampled_train_trajs_coorgrid, traj_id_to_ori_id, max_distance
            )
            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # the main loop meets this and does a test set evaluation

    def trajsimi_dataset_generator_pairs_batchi_baseline(self, trajs_merc, max_distance, last_sampled_indices=None, iteration_counts=iteration_counts):
        """
            sampled_indices: all trajectory data ids
            batch_traj_ids: trajectory data organized by batch
            batch_matrix_simis: similarity matrix organized by batch
            batch_traj_ids_top_k: topk trajectory data ids organized by batch
        """
        sampled_indices, batch_traj_ids, batch_matrix_simis = build_training_samples(self, trajs_merc, None, Config.trajsimi_measure, counts, \
                                                                                     tmn_batch_size_testing, strtegy=Config.strategy)
        
        if Config.strategy not in ['curriculum', 'dynamic_random']:
            random.shuffle(sampled_indices)
        self.last_sampled_indices = sampled_indices

        # preprocess the sampled trajectory data
        sampled_train_traj = [trajs_merc[idx] for idx in sampled_indices]
        sampled_train_trajs_coorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(sampled_train_traj)
        traj_id_to_ori_id = {idx: i for i, idx in enumerate(sampled_indices)}

        batch_count = 0
        while batch_count < iteration_counts * len(batch_traj_ids):
            batch_idx = batch_count%len(batch_traj_ids)#random.choice(range(len(batch_traj_ids)))
            batch_traj_id = batch_traj_ids[batch_idx]
            batch_train_sim = batch_matrix_simis[batch_idx]
            
            yield from self._generate_standard_batch(
                    batch_traj_id, batch_train_sim,
                    sampled_train_trajs_coorgrid, traj_id_to_ori_id,max_distance
                )
            batch_count += 1
            if batch_count % iteration_counts == 0:
                yield 'EVAL'  # the main loop meets this and does a test set evaluation



    def _generate_standard_batch(self, batch_traj_id, datasets_simi, trajs_coorgrid, traj_id_to_ori_id, max_distance):
        len_datasets = len(batch_traj_id)
        # trajs_len = [len(traj) for traj in trajs_coorgrid]
        if max_distance < np.max(datasets_simi):
            max_distance = np.max(datasets_simi)
        sub_simi = torch.tensor(datasets_simi, 
                              device=self.device, dtype=torch.float) / max_distance

        sub_trajs_idxs = torch.tensor(batch_traj_id, device=self.device, dtype=torch.long)
        anchor_input, anchor_input_len = [], []
        for anchor_idx in range(len_datasets):
            anchor_input.append(trajs_coorgrid[traj_id_to_ori_id[batch_traj_id[anchor_idx]]])
            anchor_input_len.append(len(trajs_coorgrid[traj_id_to_ori_id[batch_traj_id[anchor_idx]]]))
        max_anchor_length = max(anchor_input_len)
        anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)

        yield np.array(anchor_input), anchor_input_len, sub_simi, sub_trajs_idxs

    def load_trajsimi_dataset(self):
        # 1. read the TrajSimi dataset
        # 2. convert merc_seq to cell_id_seqs
        dic_dataset = load_trajs(Config, train_split_ratio, eval_split_ratio, eval_num, test_num)
        eval_simis = dic_dataset['eval_simis']
        test_simis = dic_dataset['test_simis']
        max_distance = dic_dataset['max_distance']
        self.__initial_mean_std(dic_dataset['eval_trajs_merc']+dic_dataset['test_trajs_merc'])
        evals_trajcoorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(dic_dataset['eval_trajs_merc'])
        tests_trajcoorgrid = self.__trajcoor_to_trajcoor_and_trajgrid(dic_dataset['test_trajs_merc'])
        train_traj = dic_dataset['train_trajs_merc']
        logging.info("trajsimi dataset sizes. (trains/evals/tests={}/{}/{})" \
                    .format(len(train_traj), len(evals_trajcoorgrid), len(tests_trajcoorgrid)))
        eval_traj = dic_dataset['eval_trajs_merc']
        return {'train_traj': train_traj, 'eval_traj': eval_traj, 'evals_trajcoorgrid': evals_trajcoorgrid, 'tests_trajcoorgrid': tests_trajcoorgrid, \
                'evals_simi': eval_simis, 'tests_simi': test_simis, \
                'max_distance': max_distance}

    def __initial_mean_std(self, lst_trajs: list):
        xs, ys = zip(*[[p[0], p[1]] for traj in lst_trajs for p in traj])
        meanx, meany, stdx, stdy = np.mean(xs), np.mean(ys), np.std(xs), np.std(ys)
        self.meanx, self.meany, self.stdx, self.stdy = meanx, meany, stdx, stdy
        # return meanx, meany, stdx, stdy
    
    def __trajcoor_to_trajcoor_and_trajgrid(self, lst_trajs: list):
        # lst_trajs = [ [[lon, lat_in_merc_space] ,[] ], ..., [], ..., ] 

        # logging.info("trajcoor_to_trajgrid starts. #={}".format(len(lst_trajs))) # for debug
        _time = time.time()

        lst_trajs_nodes_gridid = []
        for traj in lst_trajs:
            traj_nodes_gridid = []
            for xy in traj:
                i_x = int( (xy[0] - self.cellspace.x_min) / self.cellspace.x_unit )
                i_y = int( (xy[1] - self.cellspace.y_min) / self.cellspace.y_unit )
                i_x = i_x - 1 if i_x == self.cellspace.x_size else i_x
                i_y = i_y - 1 if i_y == self.cellspace.y_size else i_y
                traj_nodes_gridid.append( (i_x, i_y) )
            lst_trajs_nodes_gridid.append( traj_nodes_gridid )       

        # local normalization
        assert self.meanx is not None and self.meany is not None and self.stdx is not None and self.stdy is not None
        
        lst_trajs_nodes_lonlat = [[[(p[0] -self.meanx) / self.stdx, (p[1] - self.meany) / self.stdy] for p in traj] for traj in lst_trajs]

        lst_trajs_node_lonlat_gridid = []
        for i in range(len(lst_trajs_nodes_lonlat)):
            traj = []
            for coor, grid in zip(lst_trajs_nodes_lonlat[i], lst_trajs_nodes_gridid[i]):
                traj.append([coor[0], coor[1], grid[0], grid[1]])
            lst_trajs_node_lonlat_gridid.append(traj)  
              

        # logging.info("trajcoor_to_trajgrid ends. @={:.3f}, #={}" \
                    # .format(time.time() - _time, len(lst_trajs_node_lonlat_gridid))) # for debug
        # lst_trajs_nodes_lonlat = [ [[lon, lat_normalized] ,[] ], ..., [], ..., ] 
        # lst_trajs_nodes_gridid = [ [gridid, gridid, ... ], ..., [], ..., ] 
        # return lst_trajs_nodes_lonlat, lst_trajs_nodes_gridid # for debug
        return lst_trajs_node_lonlat_gridid



class WeightedRankingLoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)

    def forward(self, p_input, p_target, n_input, n_target):
        # trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).to(Config.device), False)
        trajs_mse_loss = self.positive_loss(p_input, p_target, False)

        # negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).to(Config.device), True)
        negative_mse_loss = self.negative_loss(n_input, n_target, True)

        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss,negative_mse_loss])
        return loss


class WeightMSELoss(nn.Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        weight_lst = [0] + list(range(sampling_num, 0, -1))
        self.register_buffer('weight', torch.tensor(weight_lst, dtype = torch.float, requires_grad = False))

        # self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU = False):
        div = target.squeeze(-1) - input
        if isReLU:
            div = F.relu(div)
        square = torch.mul(div, div)
        weight = self.weight.repeat(square.shape[0] // self.weight.shape[0])
        weight_square = torch.mul(square, weight / weight.sum())
        loss = torch.sum(weight_square)
        return loss


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
    parser.add_argument('--test', type = str, default="false", required=False, help = '')
    parser.add_argument('--gpu_id', type = int, help = '')
   
    # Meta Learning related parameters
    parser.add_argument('--use_meta_learning', action='store_true', 
                       help='enable meta learning, use the validation set to assist in correcting the training set weights')
    parser.add_argument('--cluster_num', type=int, default=10, 
                       help='number of clusters (default: 10)')
    parser.add_argument('--num_iterations', type=int, default=1000, 
                       help='number of MCTS iterations (default: 1000)')
    parser.add_argument('--budget', type=int, default=1000000, 
                       help='budget for MCTS (default: 1000000)')
    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


# nohup python TMN_trajsimi_train.py --dataset xian --trajsimi_measure dtw --seed 2000 --debug &> ../result &
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

    tmn = TMNTrainer()
    if Config.test is not None and Config.test.lower() == 'true':
        tmn.test()
    else:
        tmn.train()
