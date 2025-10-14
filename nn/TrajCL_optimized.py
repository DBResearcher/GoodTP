
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 1601):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * torch.log(torch.tensor(10000.0)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros(maxlen, emb_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TrajCL(nn.Module):
    def __init__(self, cellspace, ninput, nhidden=1024, nhead=4, nlayer=1, 
                 attn_dropout=0.1, pos_dropout=0.1, use_spatial_attn=True):
        super(TrajCL, self).__init__()
        self.cellspace = cellspace
        self.embs = nn.Parameter(data=torch.randn(self.cellspace.size(), ninput, dtype=torch.float32), requires_grad=True)
        
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.use_spatial_attn = use_spatial_attn
        
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)
        trans_encoder_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = nn.TransformerEncoder(trans_encoder_layers, nlayer)
        
        if use_spatial_attn:
            # 简化的空间注意力 - 减少层数和隐藏维度
            self.spatial_attn = SpatialAttentionEncoder(4, 16, 1, 1, attn_dropout, pos_dropout)
            self.gamma_param = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, src, attn_mask, src_padding_mask, src_len, srcspatial):
        # src: [seq_len, batch_size, emb_size]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]
        # srcspatial: [seq_len, batch_size, 4]

        if self.use_spatial_attn and srcspatial is not None:
            _, attn_spatial = self.spatial_attn(srcspatial, attn_mask, src_padding_mask, src_len)
            # 保持与原始代码相同的维度处理
            attn_spatial = attn_spatial.repeat(self.nhead, 1, 1)
            gamma = torch.sigmoid(self.gamma_param) * 10
            attn_spatial = gamma * attn_spatial
        else:
            attn_spatial = None

        src = self.pos_encoder(src)
        rtn = self.trans_encoder(src, attn_spatial, src_padding_mask)

        # 优化的池化操作
        mask = (~src_padding_mask.T).float().unsqueeze(-1)
        rtn = (rtn * mask).sum(0) / src_len.unsqueeze(-1)

        return rtn

    @torch.no_grad()
    def interpret(self, inputs):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_emb_p, trajs1_len = inputs
        trajs1_emb = trajs1_emb.to(device)
        trajs1_emb_p = trajs1_emb_p.to(device)
        trajs1_len = trajs1_len.to(device)
        max_trajs1_len = trajs1_len.max().item()
        src_padding_mask1 = torch.arange(max_trajs1_len, device=device)[None, :] >= trajs1_len[:, None]
        traj_embs = self.forward(trajs1_emb, None, src_padding_mask1, trajs1_len, trajs1_emb_p)
        return traj_embs

    @torch.no_grad()
    def trajsimi_interpret(self, inputs1, inputs2):
        device = next(self.parameters()).device
        trajs1_emb, trajs1_emb_p, trajs1_len = inputs1
        trajs2_emb, trajs2_emb_p, trajs2_len = inputs2
        
        trajs1_emb = trajs1_emb.to(device)
        trajs1_emb_p = trajs1_emb_p.to(device)
        trajs1_len = trajs1_len.to(device)
        
        trajs2_emb = trajs2_emb.to(device)
        trajs2_emb_p = trajs2_emb_p.to(device)
        trajs2_len = trajs2_len.to(device)
        
        max_trajs1_len = trajs1_len.max().item()
        src_padding_mask1 = torch.arange(max_trajs1_len, device=device)[None, :] >= trajs1_len[:, None]
        max_trajs2_len = trajs2_len.max().item()
        src_padding_mask2 = torch.arange(max_trajs2_len, device=device)[None, :] >= trajs2_len[:, None]
        
        traj_embs = self.forward(trajs1_emb, None, src_padding_mask1, 
                                 trajs1_len, trajs1_emb_p)
        traj_embs2 = self.forward(trajs2_emb, None, src_padding_mask2, 
                                  trajs2_len, trajs2_emb_p)

        dists = F.pairwise_distance(traj_embs, traj_embs2, p=1)
        return dists.detach().cpu().tolist()

class SpatialAttentionEncoder(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_dropout):
        super(SpatialAttentionEncoder, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)
        trans_encoder_layers = MyTransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = MyTransformerEncoder(trans_encoder_layers, nlayer)
        
    def forward(self, src, attn_mask, src_padding_mask, src_len):
        src = self.pos_encoder(src)
        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)

        # 优化的池化操作
        mask = (~src_padding_mask.T).float().unsqueeze(-1)
        rtn = (rtn * mask).sum(0) / src_len.unsqueeze(-1)

        return rtn, attn

class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        attn = None

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 减少前馈网络维度
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn

# 辅助函数，保持与原始代码兼容
def input_processing(trajs, cellspace, embs):
    from nn.utils.traj import merc2cell2, generate_spatial_features
    trajs2_cell, trajs2_p = zip(*[merc2cell2(t, cellspace) for t in trajs])
    trajs2_emb_p = [torch.tensor(generate_spatial_features(t, cellspace), dtype=torch.float32) for t in trajs2_p]
    trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first=False)

    trajs2_emb_cell = [embs[list(t)] for t in trajs2_cell]
    trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first=False)

    trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype=torch.long)
    
    return trajs2_emb_cell, trajs2_emb_p, trajs2_len

def collate_fn(batch, cellspace, embs):
    src, src2 = zip(*batch)
    inputs = input_processing(src, cellspace, embs)
    inputs2 = input_processing(src2, cellspace, embs)
    return inputs, inputs2

def collate_fn_single(src, cellspace, embs):
    inputs = input_processing(src, cellspace, embs)
    return inputs

class TrajCLMoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k):
        super(TrajCLMoCo, self).__init__()
        
        from nn.moco import MoCo
        seq_emb_dim = encoder_q.ninput

        self.clmodel = MoCo(encoder_q, encoder_k, 
                            seq_emb_dim,
                            seq_emb_dim // 2, 
                            2048,
                            temperature=0.05)

    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len):
        device = next(self.parameters()).device
        
        max_trajs1_len = trajs1_len.max().item()
        max_trajs2_len = trajs2_len.max().item()
        src_padding_mask1 = torch.arange(max_trajs1_len, device=device)[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = torch.arange(max_trajs2_len, device=device)[None, :] >= trajs2_len[:, None]
        
        logits, targets = self.clmodel({'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},  
                {'src': trajs2_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p})
        return logits, targets

    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)

# 性能测试函数
def benchmark_model_performance(model, batch_size=32, seq_len=100, device='cuda'):
    """测试模型性能"""
    import time
    
    # 创建测试数据
    src = torch.randn(seq_len, batch_size, model.ninput).to(device)
    src_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    src_len = torch.randint(seq_len//2, seq_len, (batch_size,)).to(device)
    srcspatial = torch.randn(seq_len, batch_size, 4).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(src, None, src_padding_mask, src_len, srcspatial)
    
    # 测试推理时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(src, None, src_padding_mask, src_len, srcspatial)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"吞吐量: {batch_size/avg_time:.2f} samples/sec")
    
    return avg_time 