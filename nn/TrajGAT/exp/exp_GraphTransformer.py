import copy
import time
import datetime
from tqdm import tqdm
import logging
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
import dgl


from exp.exp_basic import ExpBasic
from model.network.graph_transformer import GraphTransformer

from utils.build_qtree import build_qtree
from utils.pre_embedding import get_pre_embedding

from utils.data_loader import TrajGraphDataLoader
from model.loss import WeightedRankingLoss
from model.accuracy_functions import get_embedding_acc

from utils.tools import pload, pdump


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    logging.info("MODEL Total parameters:{}\n".format(total_param))
    return total_param


class ExpGraphTransformer(ExpBasic):
    def __init__(self, config, gpu_id, load_model, just_embeddings):
        self.load_model = load_model
        self.store_embeddings = just_embeddings

        super(ExpGraphTransformer, self).__init__(config, gpu_id)

        if just_embeddings:  # 只进行embedding操作
            self.qtree = build_qtree(pload(self.config["traj_path"]), 
                                    self.config["x_range"], self.config["y_range"], 
                                    self.config["max_nodes"], self.config["max_depth"])
            # 决定是否要进行 embedding预训练
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])
            self.embeding_loader = self._get_dataloader(flag="embed")
            logging.info("Embedding Graphs: {}".format(len(self.embeding_loader.dataset)))
        else:
            # self.log_writer = SummaryWriter(f"./runs/{self.config['data']}/{self.config['length']}/{self.config['model']}_{self.config['dis_type']}_{datetime.datetime.now()}/")

            logging.info("[!] Build qtree, max nodes:" + str(self.config["max_nodes"]) \
                                    + "max depth:" + str(self.config["max_depth"]) \
                                    + "x_range:" + str(self.config["x_range"]) \
                                    + "y_range:" + str(self.config["y_range"]))

            _timestamp = time.time()
            # self.qtree = build_qtree(pload(self.config["traj_path"]),  # TODO
            self.qtree = build_qtree(pload(self.config["traj_path"])[self.config["train_data_range"][0] : self.config["train_data_range"][1]], 
                                     self.config["x_range"], self.config["y_range"], 
                                     self.config["max_nodes"], self.config["max_depth"])
            logging.info("[build_qtree] done {:.4f}".format( time.time() - _timestamp ))
            
            # 进行embedding预训练
            _timestamp = time.time()
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])
            logging.info("[pre_embedding] done {:.4f}".format( time.time() - _timestamp ))

            self.train_loader = self._get_dataloader(flag="train")
            logging.info("Training Graphs: {}".format(len(self.train_loader.dataset)))

            self.val_loader = self._get_dataloader(flag="val")
            logging.info("Validation Graphs: {}".format(len(self.val_loader.dataset)))

            self.test_loader = self._get_dataloader(flag="test")
            logging.info("Validation Graphs: {}".format(len(self.test_loader.dataset)))

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.config["model"] == "TrajGAT":
            model = GraphTransformer(d_input=self.config["d_input"], \
                                    d_model=self.config["d_model"], \
                                    num_head=self.config["num_head"], \
                                    num_encoder_layers=self.config["num_encoder_layers"], \
                                    d_lap_pos=self.config["d_lap_pos"], \
                                    encoder_dropout=self.config["encoder_dropout"], \
                                    layer_norm=self.config["layer_norm"], \
                                    batch_norm=self.config["batch_norm"], \
                                    in_feat_dropout=self.config["in_feat_dropout"], \
                                    pre_embedding=self.pre_embedding,
                                    qtree=self.qtree,
                                    qtree_name2id=self.qtree_name2id,
                                    x_range=self.config["x_range"],
                                    y_range=self.config["y_range"])  # 预训练得到的，每个结点的 structure embedding

        view_model_param(model)

        if self.load_model is not None:
            model.load_state_dict(torch.load(self.load_model))
            logging.info("[!] Load model weight:{}".format(self.load_model))

        return model

    def _get_dataloader(self, flag):
        if flag == "train":
            trajs = pload(self.config["traj_path"])[self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            logging.info("Train traj number:{}".format(len(trajs)))
            matrix = pload(self.config["dis_matrix_path"])[self.config["train_data_range"][0] : self.config["train_data_range"][1], self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            # logging.info("Train matrix shape:", matrix.shape)
            # logging.info(matrix[:5, :5])
        elif flag == "val":
            # trajs = pload(self.config["traj_path"]) # orginal version
            trajs = pload(self.config["traj_path"])[self.config["val_data_range"][0] : self.config["val_data_range"][1]]
            logging.info("Val traj number:{}".format(len(trajs)))
            matrix = pload(self.config["dis_matrix_path"])[self.config["val_data_range"][0] : self.config["val_data_range"][1], self.config["val_data_range"][0] : self.config["val_data_range"][1]]
            # logging.info("Val matrix shape:", matrix.shape)
            # logging.info(matrix[:5, :5])
            # logging.info(matrix[:, 6000:10000])
        elif flag == "test":
            trajs = pload(self.config["traj_path"])[self.config["test_data_range"][0] : self.config["test_data_range"][1]]
            logging.info("Test traj number:{}".format(len(trajs)))
            matrix = pload(self.config["dis_matrix_path"])[self.config["test_data_range"][0] : self.config["test_data_range"][1], self.config["test_data_range"][0] : self.config["test_data_range"][1]]

        elif flag == "embed":
            trajs = pload(self.config["traj_path"])
            matrix = pload(self.config["dis_matrix_path"])

        data_loader = TrajGraphDataLoader(traj_data=trajs, dis_matrix=matrix, 
                                    phase=flag, train_batch_size=self.config["train_batch_size"], 
                                    eval_batch_size=self.config["eval_batch_size"], 
                                    d_lap_pos=self.config["d_lap_pos"], sample_num=self.config["sample_num"], 
                                    num_workers=self.config["num_workers"], data_features=self.config["data_features"], 
                                    x_range=self.config["x_range"], y_range=self.config["y_range"], 
                                    qtree=self.qtree, qtree_name2id=self.qtree_name2id).get_data_loader()

        return data_loader

    def _select_optimizer(self):
        if self.config["optimizer"] == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.config["init_lr"])
        elif self.config["optimizer"] == "Adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.config["init_lr"])

        return model_optim, None

    def _select_criterion(self):
        criterion = WeightedRankingLoss(self.config["sample_num"], self.config["alpha"], self.device).float()
        return criterion

    def embedding(self):
        all_vectors = []
        self.model.eval()

        loader_time = 0
        begin_time = time.time()
        mark_time = time.time()
        for trajgraph_l_l, _ in tqdm(self.embeding_loader, mininterval = 20.0):
            loader_time += time.time() - mark_time
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            for b in trajgraph_l_l:
                traj_graph.extend(b)
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]

            all_vectors.append(vectors)
            mark_time = time.time()

        all_vectors = torch.cat(all_vectors, dim=0)
        logging.info("all_embeding_vectors length:{}".format(len(all_vectors)))
        logging.info("all_embedding_vectors shape:{}".format(all_vectors.shape))

        end_time = time.time()
        logging.info(f"all embedding time: {end_time-begin_time-loader_time} seconds")

        # pdump(all_vectors, f"{self.config['length']}_{self.config['dis_type']}_embeddings_{all_vectors.shape[0]}_{all_vectors.shape[1]}.pkl")

        hr5, hr20, r5_20 = get_embedding_acc(row_embedding_tensor=all_vectors, 
                                            col_embedding_tensor=all_vectors, 
                                            distance_matrix=self.embeding_loader.dataset.dis_matrix, 
                                            matrix_cal_batch=self.config["matrix_cal_batch"],)

        logging.info("hr5={} hr20={} r5_20={}".format(hr5, hr20, r5_20))

    def val(self, dataloader):
        all_vectors = []
        self.model.eval()

        for trajgraph_l_l, _ in dataloader:
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            for b in trajgraph_l_l:
                traj_graph.extend(b)
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]

            all_vectors.append(vectors)

        all_vectors = torch.cat(all_vectors, dim=0)
        logging.info("all_val_vectors length:{}".format(len(all_vectors)))

        hr5, hr20, r5_20 = get_embedding_acc(row_embedding_tensor=all_vectors, \
                                                col_embedding_tensor=all_vectors, \
                                                distance_matrix=dataloader.dataset.dis_matrix, \
                                                matrix_cal_batch=self.config["matrix_cal_batch"],)

        return hr5, hr20, r5_20

    def train(self):
        logging.info("exp_GraphTransformer train starts.")

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hr5 = 0.0
        time_now = time.time()

        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.config["epoch"]):
            self.model.train()

            epoch_begin_time = time.time()
            epoch_loss = 0.0

            dataload_time = 0
            embed_time = 0
            groupdata_time = 0
            test_time = time.time()
            i_batch = 0
            for trajgraph_l_l, dis_l in self.train_loader:
                dataload_time += time.time() - test_time
                test_time2 = time.time()
                # trajgraph_l_l [B, SAM, graph]
                # dis_l [B, SAM]
                B = len(trajgraph_l_l)
                SAM = self.config["sample_num"]
                D = self.config["d_model"]

                traj_graph = []
                for b in trajgraph_l_l:
                    traj_graph.extend(b)
                batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B*SAM, graph)
                groupdata_time += time.time() - test_time2
                test_time3 = time.time()
                model_optim.zero_grad()

                with torch.set_grad_enabled(True):
                    vectors = self.model(batch_graphs)  # vecters [B*SAM, d_model]

                vectors = vectors.view(B, SAM, D)

                loss = criterion(vectors, torch.tensor(dis_l).to(self.device))

                loss.backward()
                model_optim.step()

                epoch_loss += loss.item()
                embed_time += time.time() - test_time3
                test_time = time.time()
                
                if i_batch % 50 == 0:
                    logging.info("Epoch {} batch {} end. loss={:.5f}, acc_time={:.1f}".format( \
                                    epoch, i_batch, loss.item(), time.time()-epoch_begin_time))
                i_batch += 1

            logging.info("\nLoad data time:{}m".format(dataload_time // 60))
            logging.info("Data group time:{}m".format(groupdata_time // 60))
            logging.info("Train model time:{}m\n".format(embed_time // 60))

            epoch_loss = epoch_loss / len(self.train_loader.dataset)
            # self.log_writer.add_scalar(f"TrajRepresentation/Loss", float(epoch_loss), epoch)

            # scheduler.step(epoch_loss)

            logging.info("Epoch {} train done. loss={:.4f}, time={:.4f}".format(epoch, epoch_loss, time.time()-epoch_begin_time))

            val_begin_time = time.time()
            hr5, hr20, r5_20 = self.val(self.val_loader)
            val_end_time = time.time()

            # self.log_writer.add_scalar(f"TrajRepresentation/HR5", hr5, epoch)
            # self.log_writer.add_scalar(f"TrajRepresentation/HR20", hr20, epoch)
            # self.log_writer.add_scalar(f"TrajRepresentation/R5@20", r5_20, epoch)

            logging.info(f"Val HR5: {100 * hr5:.4f}%\tHR20: {100 * hr20:.4f}%\tR5@20: {100 * r5_20:.4f}%\tTime: {(val_end_time -val_begin_time) // 60} m {int((val_end_time -val_begin_time) % 60)} s")

            if hr5 > best_hr5:
                best_hr5 = hr5
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_end = time.time()

        logging.info("\nAll training complete in {:.4f}s".format(time_end - time_now))
        logging.info(f"Best HR5: {100*best_hr5:.4f}%")

        # cp_file = self.config["model_best_wts_path"].format(self.config["data"], self.config["length"], self.config["model"], self.config["dis_type"], best_hr10)
        # torch.save(best_model_wts, cp_file)

        checkpoint_filepath = 'model/wts/{}_{}_{}'.format(self.config["data"], self.config["model"], self.config["d_model"]) 
        # checkpoint_filepath = 'model/wts/varying_qtree_node_capacity/{}_{}_{}'.format(self.config["data"], self.config["model"], self.config["max_nodes"])
        torch.save({"encoder" : self.model.state_dict(),
                    "qtree": self.model.qtree,
                    "qtree_name2id": self.model.qtree_name2id,
                    "x_range": self.model.x_range,
                    "y_range": self.model.y_range       }, checkpoint_filepath)
        logging.info("checkpoint saved!")

        self.model.load_state_dict(best_model_wts)
        self.model.to(self.device)
        
        self.trajsimi_exp()
        
        return


    def trajsimi_exp(self):
        # load testing data
        # call eval()
        torch.cuda.empty_cache()

        _ts = time.time()
        logging.info("[trajsimi_exp] start.")

        metrics = self.val(self.test_loader)
        
        logging.info("[trajsimi_exp] end. @={:.4f}".format( time.time() - _ts ))
        logging.info('[EXPFlag]model=TrajGAT,dataset={},fn={},hr5={},hr20={},hr20in5={}'.format( \
                    self.config['data'], self.config['dis_type'], metrics[0], metrics[1], metrics[2]))
        return