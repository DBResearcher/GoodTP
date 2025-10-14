
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import dgl.function as fn
import dgl
from .TrajGAT_utils import trajlist_to_trajgraph
#========================================================================

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # 改进数值稳定性：使用更严格的裁剪和检查
        score = edges.data[field] / scale_constant
        
        # 检查并处理 NaN 和 Inf
        if torch.isnan(score).any() or torch.isinf(score).any():
            # 如果包含 NaN 或 Inf，设置为较小的值
            score = torch.where(torch.isnan(score) | torch.isinf(score), 
                              torch.tensor(-10.0, device=score.device), score)
        
        # 使用更严格的裁剪范围
        score = torch.clamp(score, -10, 10)
        
        return {field: torch.exp(score)}
    return func


def message_func(edges):
    # message UDF for equation (3) & (4)
    return {"V_h": edges.src["V_h"], "score": edges.data["score"]}


def reduce_func(nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3) - 改进 softmax 的数值稳定性
    scores = nodes.mailbox["score"]
    
    # 检查分数是否包含 NaN 或 Inf
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        # 如果包含 NaN 或 Inf，使用均匀分布
        alpha = torch.ones_like(scores) / scores.size(1)
    else:
        # 使用数值稳定的 softmax
        # 减去最大值以提高数值稳定性
        scores_max = torch.max(scores, dim=1, keepdim=True)[0]
        scores_stable = scores - scores_max
        alpha = F.softmax(scores_stable, dim=1)
    
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox["V_h"], dim=1)
    
    # 检查输出是否包含 NaN 或 Inf
    if torch.isnan(h).any() or torch.isinf(h).any():
        # 如果输出包含 NaN 或 Inf，使用输入的平均值
        h = torch.mean(nodes.mailbox["V_h"], dim=1)
    
    return {"V_h": h}

#========================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, use_bias=False):
        super(MultiHeadAttention,self).__init__()
        assert d_model % n_head == 0, "d_model must can be divisible by n_head"
        self.dim_head = d_model // n_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))
        g.apply_edges(scaled_exp("score", np.sqrt(self.dim_head)))

        g.update_all(message_func, reduce_func)

    def forward(self, g, h):
        Q_h = self.W_q(h).view(-1, self.n_head, self.dim_head)
        K_h = self.W_k(h).view(-1, self.n_head, self.dim_head)
        V_h = self.W_v(h).view(-1, self.n_head, self.dim_head)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h
        g.ndata["K_h"] = K_h
        g.ndata["V_h"] = V_h

        self.propagate_attention(g)

        head_out = g.ndata["V_h"]

        return head_out #【node num, n_head, d_head】


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, layer_norm, batch_norm):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.W_o = nn.Linear(d_model, d_model)

        if batch_norm:
            self.norm_part1 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part1 = nn.LayerNorm(d_model)

        # FFN
        self.FFN_layer1 = nn.Linear(d_model, d_model * 2)
        self.FFN_layer2 = nn.Linear(d_model * 2, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if batch_norm:
            self.norm_part2 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part2 = nn.LayerNorm(d_model)

    def forward(self, g, h):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.d_model)

        h = self.dropout1(h)
        h = self.W_o(h)
        h = self.norm_part1(h_in1 + h)  # residual connection & normalization

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.dropout2(h)
        h = self.FFN_layer2(h)

        h = self.norm_part2(h + h_in2)

        return h


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, traj_transformer):
        super(Encoder, self).__init__()
        self.encoder = _get_clones(encoder_layer, num_layers)
        self.traj_transformer = traj_transformer

    def forward(self, g: dgl.DGLGraph, h):
        for layer in self.encoder:
            h = layer(g, h)
        g.ndata["h"] = h

        # vectors = dgl.readout_nodes(g, "h", op="mean")
        # Instead of dgl.readout_nodes, use a transformer encoder for each trajectory
        batch_num_nodes = g.batch_num_nodes().tolist() if hasattr(g, 'batch_num_nodes') else g.batch_num_nodes().numpy().tolist()
        h = g.ndata["h"]
        vectors = []
        start = 0
        # 将所有轨迹padding到同一长度，批量送入transformer并行处理
        max_len = max(batch_num_nodes)
        padded_trajs = []
        mask = []
        for n in batch_num_nodes:
            traj_h = h[start:start+n]  # [seq_len, d_model]
            pad_len = max_len - n
            if pad_len > 0:
                pad = torch.zeros((pad_len, traj_h.shape[1]), dtype=traj_h.dtype, device=traj_h.device)
                traj_h = torch.cat([traj_h, pad], dim=0)
            padded_trajs.append(traj_h.unsqueeze(1))  # [max_len, 1, d_model]
            # mask: 1 for real, 0 for pad
            mask.append(torch.cat([torch.ones(n, device=traj_h.device), torch.zeros(pad_len, device=traj_h.device)]))
            start += n
        padded_trajs = torch.cat(padded_trajs, dim=1)  # [max_len, batch_size, d_model]
        mask = torch.stack(mask, dim=1)  # [max_len, batch_size]

        # transformer支持mask，mask==0的位置会被忽略
        # 这里假设self.traj_transformer支持src_key_padding_mask参数
        # src_key_padding_mask: [batch_size, max_len], 1表示pad, 0表示有效
        # 但pytorch transformer是1表示pad, 0表示有效
        src_key_padding_mask = (mask == 0).transpose(0, 1)  # [batch_size, max_len]
        traj_h_trans = self.traj_transformer(padded_trajs, src_key_padding_mask=src_key_padding_mask)  # [max_len, batch_size, d_model]
        # 取每个轨迹的第一个token作为embedding
        traj_embs = traj_h_trans[0]  # [batch_size, d_model]
        vectors = [traj_embs[i:i+1] for i in range(traj_embs.shape[0])]
        vectors = torch.cat(vectors, dim=0)

        # # 只选取 真实节点 进行mean
        # all_feat = g.ndata["h"]
        # all_flag = g.ndata["flag"]
        # print(all_flag[:600])
        # vectors = None

        return vectors  # [graph num, d_model]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#========================================================================

class GraphTransformer(nn.Module):
    def __init__(self, d_input, d_model, num_head, num_encoder_layers, d_lap_pos, 
                encoder_dropout, pre_embedding, qtree, qtree_name2id,
                x_range, y_range,
                layer_norm=False, batch_norm=True, in_feat_dropout=0.0):
        super(GraphTransformer, self).__init__()
        self.embedding_h = nn.Linear(d_input, d_model)
        self.embedding_lap_pos = nn.Linear(d_lap_pos, d_model)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.d_model = d_model
        self.num_head = num_head
        self.encoder_dropout = encoder_dropout

        # embedding layer for each node
        if pre_embedding is not None:
            # self.embedding_id = nn.Embedding.from_pretrained(pre_embedding, freeze=False)
            self.embedding_id = nn.Embedding.from_pretrained(pre_embedding)  # no word embedding update

            # total_num = sum(p.numel() for p in self.embedding_id.parameters())
            # trainable_num = sum(p.numel() for p in self.embedding_id.parameters() if p.requires_grad)
            # logging.info(f"Embedding Total: {total_num}, Trainable: {trainable_num}")

            self.use_pre_embedding = True
        else:
            # 创建一个空的 embedding 层，确保模型加载时不会缺少参数
            # 使用一个小的 vocab_size，因为实际不会用到
            self.embedding_id = nn.Embedding(1, d_model)
            self.use_pre_embedding = False
            
        self.qtree = qtree
        self.qtree_name2id = qtree_name2id
        self.x_range = x_range
        self.y_range = y_range

        encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_head, dropout=encoder_dropout, layer_norm=layer_norm, batch_norm=batch_norm)
        
        # Add a per-trajectory transformer encoder
        self.traj_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_head, d_model*2, encoder_dropout),
            num_layers=1
        )

        self.encoder = Encoder(encoder_layer, num_encoder_layers, self.traj_transformer)
        self._reset_parameters()


    def forward(self, g):         
        h = g.ndata["feat"]  # num x feat

        # h_lap_pos = g.ndata["lap_pos_feat"]
        # sign_flip = torch.rand(h_lap_pos.size(1)).to(h_lap_pos.device)
        # sign_flip[sign_flip >= 0.5] = 1.0
        # sign_flip[sign_flip < 0.5] = -1.0
        # h_lap_pos = h_lap_pos * sign_flip.unsqueeze(0)

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # position embedding
        # h_lap_pos = self.embedding_lap_pos(h_lap_pos.float())

        # id embedding
        if self.use_pre_embedding:
            max_id = g.ndata['id'].max().item()
            num_embeddings = self.embedding_id.num_embeddings
            assert max_id < num_embeddings, f"Node id {max_id} out of embedding range {num_embeddings}"
            h_id = g.ndata["id"]  # pre mebedding feat
            h_id = self.embedding_id(h_id)

            # h = h + h_lap_pos + h_id
            h = h + h_id
        else:
            # h = h + h_lap_pos
            h = h

        vectors = self.encoder(g, h)  # vectors [g_num, d_model]

        return vectors

   
    @torch.no_grad()
    def interpret(self, g):
        device = next(self.parameters()).device
        embs = self.forward(g.to(device))
        return embs


    @torch.no_grad()
    def trajsimi_interpret(self, g, g2):
        device = next(self.parameters()).device
        embs = self.forward(g.to(device))
        embs2 = self.forward(g2.to(device))
        dists = F.pairwise_distance(embs, embs2, p = 1)
        return dists.detach().cpu().tolist()
    
    
    @torch.no_grad()
    def load_checkpoint(self, cp_file, device):
        cp = torch.load(cp_file, map_location = device)
        
        # 检查是否是新的保存格式（只包含 model 键）
        if 'model' in cp:
            state_dict = cp['model']
        else:
            state_dict = cp['encoder']
            
        # 检查是否有 embedding_id.weight 参数
        if 'embedding_id.weight' in state_dict:
            self.use_pre_embedding = True
            # 如果当前模型没有 embedding_id 或大小不匹配，重新创建
            if not hasattr(self, 'embedding_id') or self.embedding_id.weight.shape != state_dict['embedding_id.weight'].shape:
                vocab_size = state_dict['embedding_id.weight'].shape[0]
                embedding_dim = state_dict['embedding_id.weight'].shape[1]
                self.embedding_id = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.use_pre_embedding = False
            
        # 加载模型参数
        self.load_state_dict(state_dict, strict=False)
        
        # 加载其他参数（如果存在）
        if 'qtree' in cp:
            self.qtree = cp['qtree']
        if 'qtree_name2id' in cp:
            self.qtree_name2id = cp['qtree_name2id']
        if 'x_range' in cp:
            self.x_range = cp['x_range']
        if 'y_range' in cp:
            self.y_range = cp['y_range']
        
        # 确保 self.traj_transformer 还是 nn.Module
        self.traj_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.num_head, self.d_model*2, self.encoder_dropout),
            num_layers=1
        )
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

# for trajsimi 
def collate_fn(batch, qtree, qtree_name2id, x_range, y_range):
    src, src2 = zip(*batch)
    lst  = trajlist_to_trajgraph(src, qtree, qtree_name2id, x_range, y_range)
    g = dgl.batch(lst)
    lst2  = trajlist_to_trajgraph(src2, qtree, qtree_name2id, x_range, y_range)
    g2 = dgl.batch(lst2)
    return g, g2

# for knn
def collate_fn_single(src, qtree, qtree_name2id, x_range, y_range):
    lst  = trajlist_to_trajgraph(src, qtree, qtree_name2id, x_range, y_range)
    g = dgl.batch(lst)
    return g