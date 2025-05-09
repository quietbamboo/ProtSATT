import torch
from torch import nn
import torch.nn.functional as F
import math

# Standardization 均值方差归一化
def standardization(seq):
    seq[:] = (torch[:] - torch.mean(seq))/ torch.std(seq)
    return seq

# 最值归一化
def max_standardization(seq):
    seq = (seq-torch.min(seq))/(torch.max(seq)-torch.min(seq))
    return seq

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# ==============多层注意力=====================
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish,self).__init__()
        self.inplace=inplace
    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)

class ClippedLinearActivation(nn.Module):
    def __init__(self):
        super(ClippedLinearActivation, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=0, max=1)

class CosineNegActivation(nn.Module):
    def __init__(self):
        super(CosineNegActivation, self).__init__()
    def forward(self, x):
        # print('xxx', x)
        # if x<=0 or x>=1:
        #     return torch.clamp(x, min=0, max=1)
        # else:
        return torch.where(x < 0, torch.tensor(0.0, device=x.device), 
                           torch.where(x > 1, torch.tensor(1.0, device=x.device),
                                       (-torch.cos(math.pi * x) + 1) / 2))

class SelfAttention(nn.Module):
    def __init__(self, query_dim, return_dim, num_heads=1, dropout=0.15, encoder_with_res=True, residual_coef=1):
        super(SelfAttention, self).__init__()
        self.query_dim = query_dim
        self.return_dim = return_dim

        # Define the weights
        self.WQ = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.WK = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.WV = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.linear = nn.LazyLinear(return_dim) # 128
        self.drop = nn.Dropout(dropout)
        self.num_heads = num_heads

        self.init_weights()
        self.encoder_with_res = encoder_with_res
        self.residual_coef = residual_coef
        self.LayerNorm = LayerNorm(query_dim, eps=1e-12)
        self.swish = Swish()

    def init_weights(self):
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def forward(self, x):
        # print('========',x.shape)
        batch_size, seq_length, _ = x.size()

        # Linear projections
        Q = torch.matmul(x, self.WQ).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)
        K = torch.matmul(x, self.WK).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)
        V = torch.matmul(x, self.WV).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.query_dim // self.num_heads)
        # QK = F.softmax(QK, dim=-1)
        QK = self.swish(QK)

        output = torch.matmul(QK, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.query_dim)
        # print('===============', output.shape)
        
        # # Apply dropout if specified
        # if self.dropout:
        #     output = self.dropout(output)

        # # Apply normalization if specified
        # if self.encoder_norm:
        #     output = self.encoder_norm(output)

        # # Add residual connection if specified
        if self.encoder_with_res:
            output = self.LayerNorm(output + self.residual_coef * x)

        output = self.drop(self.linear(output.view(output.size(0), -1)))
        return self.swish(output).view(output.size(0), -1, self.query_dim)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, return_dim, num_heads=1, dropout=0.15, encoder_with_res=True, residual_coef=0.5):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.return_dim = return_dim

        # Define the weights
        self.WQ = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.WK = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.WV = nn.Parameter(torch.Tensor(query_dim, query_dim))
        self.linear = nn.LazyLinear(return_dim)
        self.drop = nn.Dropout(dropout)
        self.num_heads = num_heads

        self.init_weights()
        self.encoder_with_res = encoder_with_res
        self.residual_coef = residual_coef
        self.LayerNorm = LayerNorm(query_dim, eps=1e-12)
        self.swish = Swish()

    def init_weights(self):
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def forward(self, q, x):
        batch_size, seq_length, _ = x.size()

        # Linear projections
        Q = torch.matmul(q, self.WQ).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)
        K = torch.matmul(x, self.WK).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)
        V = torch.matmul(x, self.WV).view(batch_size, seq_length, self.num_heads, self.query_dim // self.num_heads)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.query_dim // self.num_heads)
        QK = F.softmax(QK, dim=-1)
        # QK = self.swish(QK)

        output = torch.matmul(QK, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.query_dim)
        # # Add residual connection if specified
        if self.encoder_with_res:
            output = self.LayerNorm(output + self.residual_coef * x)
        output = self.drop(self.linear(output.view(output.size(0), -1)))
        return self.swish(output)

class ProtSATT(nn.Module):
    def __init__(
            self,
            *,
            dropout,
            first_self_query_dim=32, first_self_return_dim=512, first_self_num_head=1, first_self_dropout=0.15, first_self_encoder_with_res=True, first_self_residual_coef=1,
            self_deep=1, 
            deep_self_query_dim=16, deep_self_return_dim=128, deep_self_num_head=1, deep_self_dropout=0.15, deep_self_encoder_with_res=True, deep_self_residual_coef=1,
            deep_cross_query_dim=8, deep_cross_return_dim=32, deep_cross_num_head=1, deep_cross_dropout=0.15, deep_cross_encoder_with_res=True, deep_cross_residual_coef=0.5,
            out_scores=2,
            **kwargs
    ):
        super().__init__()
        self.dropout_emb = nn.Dropout(p=dropout)
        self.input1 = nn.LazyLinear(1024)
        self.input2 = nn.LazyLinear(1024)
        self.input3 = nn.LazyLinear(1024)
        self.selfAttentionInput1 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput2 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput3 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)

        self_atten_layer1 = []
        self_atten_layer2 = []
        self_atten_layer3 = []
        for i in range(self_deep):
            self_atten_layer1.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer2.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer3.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
        self.self_atten_tower1 = nn.Sequential(*self_atten_layer1)
        self.self_atten_tower2 = nn.Sequential(*self_atten_layer2)
        self.self_atten_tower3 = nn.Sequential(*self_atten_layer3)

        self.cross_atten_tower1 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower2 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower3 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower4 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower5 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower6 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.swish = Swish()
        # self.swish = ClippedLinearActivation()
        self.cos = CosineNegActivation()
        self.dense_out = nn.LazyLinear(16)
        self.to_score = nn.Linear(16, out_scores)

    def forward(self, x_embed1, x_embed2, x_embed3, device, first_self_query_dim=32, deep_self=True, deep_self_query_dim=16, deep_cross_query_dim=8, return_intermediate=False):
        n1 = x_embed1.shape[1]
        # [batch, 1900]
        x1 = self.dropout_emb(x_embed1)
        pos_emb1 = max_standardization(torch.arange(n1))
        pos_emb1 = pos_emb1.to(device)
        x1 = x1 + pos_emb1

        # queries
        n2 = x_embed2.shape[1]
        # [batch, 1900]
        x2 = self.dropout_emb(x_embed2)
        pos_emb2 = max_standardization(torch.arange(n2))
        pos_emb2 = pos_emb2.to(device)
        x2 = x2 + pos_emb2

        # queries
        n3 = x_embed3.shape[1]
        # [batch, 1900]
        x3 = self.dropout_emb(x_embed3)
        pos_emb3 = max_standardization(torch.arange(n3))
        pos_emb3 = pos_emb3.to(device)
        x3 = x3 + pos_emb3

        # 第一层self 
        x1 = self.input1(x1).view(x1.shape[0], -1, first_self_query_dim)
        x2 = self.input2(x2).view(x2.shape[0], -1, first_self_query_dim)
        x3 = self.input3(x3).view(x3.shape[0], -1, first_self_query_dim)
        # self_flatten
        atten_state1 = self.selfAttentionInput1(x1).view(x1.shape[0], -1)
        atten_state2 = self.selfAttentionInput2(x2).view(x2.shape[0], -1)
        atten_state3 = self.selfAttentionInput3(x3).view(x3.shape[0], -1)

        # 第二层self
        if deep_self:
            atten_state1 = atten_state1.view(atten_state1.shape[0], -1, deep_self_query_dim)
            atten_state2 = atten_state2.view(atten_state2.shape[0], -1, deep_self_query_dim)
            atten_state3 = atten_state3.view(atten_state3.shape[0], -1, deep_self_query_dim)
            atten_state1 = self.self_atten_tower1(atten_state1)
            atten_state2 = self.self_atten_tower2(atten_state2)
            atten_state3 = self.self_atten_tower3(atten_state3)

        # 第三层cross
        atten_state1 = atten_state1.view(atten_state1.shape[0], -1, 8)
        atten_state2 = atten_state2.view(atten_state2.shape[0], -1, 8)
        atten_state3 = atten_state3.view(atten_state3.shape[0], -1, 8)
        atten_state_tower1 = atten_state1
        atten_state_tower2 = atten_state1
        atten_state_tower3 = atten_state2
        atten_state_tower4 = atten_state2
        atten_state_tower5 = atten_state3
        atten_state_tower6 = atten_state3

        for layer in self.cross_atten_tower1:
            atten_state_tower1 = layer(atten_state_tower1, atten_state2)
        for layer in self.cross_atten_tower2:
            atten_state_tower2 = layer(atten_state_tower2, atten_state3)
        for layer in self.cross_atten_tower3:
            atten_state_tower3 = layer(atten_state_tower3, atten_state1)
        for layer in self.cross_atten_tower4:
            atten_state_tower4 = layer(atten_state_tower4, atten_state3)
        for layer in self.cross_atten_tower5:
            atten_state_tower5 = layer(atten_state_tower5, atten_state1)
        for layer in self.cross_atten_tower6:
            atten_state_tower6 = layer(atten_state_tower6, atten_state2)

        intermediate = torch.cat((atten_state_tower1,atten_state_tower2,atten_state_tower3,atten_state_tower4,atten_state_tower5,atten_state_tower6),1)
        if return_intermediate:
            return intermediate  # 返回某一层的特征
            
        score = self.cos(self.dense_out(self.swish(intermediate)))
        score = self.to_score(score)
        return score.squeeze(1)

class multi_layer_attention_2input(nn.Module):
    def __init__(
            self,
            *,
            dropout,
            first_self_query_dim=32, first_self_return_dim=512, first_self_num_head=1, first_self_dropout=0.15, first_self_encoder_with_res=True, first_self_residual_coef=1,
            self_deep=1, 
            deep_self_query_dim=16, deep_self_return_dim=128, deep_self_num_head=1, deep_self_dropout=0.15, deep_self_encoder_with_res=True, deep_self_residual_coef=1,
            deep_cross_query_dim=8, deep_cross_return_dim=32, deep_cross_num_head=1, deep_cross_dropout=0.15, deep_cross_encoder_with_res=True, deep_cross_residual_coef=0.5,
            out_scores=2,
            **kwargs
    ):
        super().__init__()
        self.dropout_emb = nn.Dropout(p=dropout)
        self.input1 = nn.LazyLinear(1024)
        self.input2 = nn.LazyLinear(1024)
        self.selfAttentionInput1 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput2 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)

        self_atten_layer1 = []
        self_atten_layer2 = []
        for i in range(self_deep):
            self_atten_layer1.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer2.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
        self.self_atten_tower1 = nn.Sequential(*self_atten_layer1)
        self.self_atten_tower2 = nn.Sequential(*self_atten_layer2)

        self.cross_atten_tower1 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower2 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])

        self.swish = Swish()
        # self.swish = ClippedLinearActivation()
        self.cos = CosineNegActivation()
        self.dense_out = nn.LazyLinear(16)
        self.to_score = nn.Linear(16, out_scores)

    def forward(self, x_embed1, x_embed2, x_embed3, device, first_self_query_dim=32, deep_self=True, deep_self_query_dim=16, deep_cross_query_dim=8, return_intermediate=False):
        n1 = x_embed1.shape[1]
        # [batch, 1900]
        x1 = self.dropout_emb(x_embed1)
        # x1 = x_embed1
        pos_emb1 = max_standardization(torch.arange(n1))
        pos_emb1 = pos_emb1.to(device)
        # torch.Size([1, 1900, 32])
        x1 = x1 + pos_emb1

        # queries
        n2 = x_embed2.shape[1]
        # [batch, 1900]
        x2 = self.dropout_emb(x_embed2)
        # x2 = x_embed2
        pos_emb2 = max_standardization(torch.arange(n2))
        pos_emb2 = pos_emb2.to(device)
        # torch.Size([1, 1900, 32])
        # pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x2 = x2 + pos_emb2

        # 第一层self 
        x1 = self.input1(x1).view(x1.shape[0], -1, first_self_query_dim)
        x2 = self.input2(x2).view(x2.shape[0], -1, first_self_query_dim)
        # self_flatten
        atten_state1 = self.selfAttentionInput1(x1).view(x1.shape[0], -1)
        atten_state2 = self.selfAttentionInput2(x2).view(x2.shape[0], -1)

        # 第二层self
        if deep_self:
            atten_state1 = atten_state1.view(atten_state1.shape[0], -1, deep_self_query_dim)
            atten_state2 = atten_state2.view(atten_state2.shape[0], -1, deep_self_query_dim)
            atten_state1 = self.self_atten_tower1(atten_state1)
            atten_state2 = self.self_atten_tower2(atten_state2)

        # 第三层cross
        atten_state1 = atten_state1.view(atten_state1.shape[0], -1, 8)
        atten_state2 = atten_state2.view(atten_state2.shape[0], -1, 8)
        atten_state_tower1 = atten_state1
        atten_state_tower2 = atten_state2

        for layer in self.cross_atten_tower1:
            atten_state_tower1 = layer(atten_state_tower1, atten_state2)
        for layer in self.cross_atten_tower2:
            atten_state_tower2 = layer(atten_state_tower2, atten_state1)

        intermediate = torch.cat((atten_state_tower1,atten_state_tower2),1)
        if return_intermediate:
            return intermediate  # 返回某一层的特征
            
        score = self.cos(self.dense_out(self.swish(intermediate)))

        score = self.to_score(score)
        return score.squeeze(1)

class multi_layer_attention_1input(nn.Module):
    def __init__(
            self,
            *,
            dropout,
            first_self_query_dim=32, first_self_return_dim=512, first_self_num_head=1, first_self_dropout=0.15, first_self_encoder_with_res=True, first_self_residual_coef=1,
            self_deep=1, 
            deep_self_query_dim=16, deep_self_return_dim=128, deep_self_num_head=1, deep_self_dropout=0.15, deep_self_encoder_with_res=True, deep_self_residual_coef=1,
            deep_cross_query_dim=8, deep_cross_return_dim=32, deep_cross_num_head=1, deep_cross_dropout=0.15, deep_cross_encoder_with_res=True, deep_cross_residual_coef=0.5,
            out_scores=2,
            **kwargs
    ):
        super().__init__()
        self.dropout_emb = nn.Dropout(p=dropout)
        self.input1 = nn.LazyLinear(1024)
        self.selfAttentionInput1 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)

        self_atten_layer1 = []
        for i in range(self_deep):
            self_atten_layer1.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
        self.self_atten_tower1 = nn.Sequential(*self_atten_layer1)

        self.swish = Swish()
        # self.swish = ClippedLinearActivation()
        self.cos = CosineNegActivation()
        self.dense_out = nn.LazyLinear(16)
        self.to_score = nn.Linear(16, out_scores)

    def forward(self, x_embed1, x_embed2, x_embed3, device, first_self_query_dim=32, deep_self=True, deep_self_query_dim=16, deep_cross_query_dim=8, return_intermediate=False):
        n1 = x_embed1.shape[1]
        # [batch, 1900]
        x1 = self.dropout_emb(x_embed1)
        # x1 = x_embed1
        pos_emb1 = max_standardization(torch.arange(n1))
        pos_emb1 = pos_emb1.to(device)
        x1 = x1 + pos_emb1

        # 第一层self 
        x1 = self.input1(x1).view(x1.shape[0], -1, first_self_query_dim)
        # self_flatten
        atten_state1 = self.selfAttentionInput1(x1).view(x1.shape[0], -1)
        # 第二层self
        if deep_self:
            atten_state1 = atten_state1.view(atten_state1.shape[0], -1, deep_self_query_dim)
            atten_state1 = self.self_atten_tower1(atten_state1)

        intermediate = atten_state1

        if return_intermediate:
            return intermediate  # 返回某一层的特征
            
        score = self.cos(self.dense_out(self.swish(intermediate)))
        score = self.to_score(score)
        return score.squeeze(1)


class multi_layer_attention_no_self(nn.Module):
    def __init__(
            self,
            *,
            dropout,
            first_self_query_dim=32, first_self_return_dim=512, first_self_num_head=1, first_self_dropout=0.15, first_self_encoder_with_res=True, first_self_residual_coef=1,
            self_deep=1, 
            deep_self_query_dim=16, deep_self_return_dim=128, deep_self_num_head=1, deep_self_dropout=0.15, deep_self_encoder_with_res=True, deep_self_residual_coef=1,
            deep_cross_query_dim=8, deep_cross_return_dim=32, deep_cross_num_head=1, deep_cross_dropout=0.15, deep_cross_encoder_with_res=True, deep_cross_residual_coef=0.5,
            out_scores=2,
            **kwargs
    ):
        super().__init__()
        self.dropout_emb = nn.Dropout(p=dropout)
        self.input1 = nn.LazyLinear(1024)
        self.input2 = nn.LazyLinear(1024)
        self.input3 = nn.LazyLinear(1024)
        self.selfAttentionInput1 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput2 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput3 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)

        self_atten_layer1 = []
        self_atten_layer2 = []
        self_atten_layer3 = []
        for i in range(self_deep):
            self_atten_layer1.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer2.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer3.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
        self.self_atten_tower1 = nn.Sequential(*self_atten_layer1)
        self.self_atten_tower2 = nn.Sequential(*self_atten_layer2)
        self.self_atten_tower3 = nn.Sequential(*self_atten_layer3)

        self.cross_atten_tower1 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower2 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower3 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower4 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower5 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower6 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.swish = Swish()
        # self.swish = ClippedLinearActivation()
        self.cos = CosineNegActivation()
        self.dense_out = nn.LazyLinear(16)
        self.to_score = nn.Linear(16, out_scores)

    def forward(self, x_embed1, x_embed2, x_embed3, device, first_self_query_dim=32, deep_self=True, deep_self_query_dim=16, deep_cross_query_dim=8, return_intermediate=False):
        n1 = x_embed1.shape[1]
        # [batch, 1900]
        x1 = self.dropout_emb(x_embed1)
        pos_emb1 = max_standardization(torch.arange(n1))
        pos_emb1 = pos_emb1.to(device)
        x1 = x1 + pos_emb1

        # queries
        n2 = x_embed2.shape[1]
        # [batch, 1900]
        x2 = self.dropout_emb(x_embed2)
        pos_emb2 = max_standardization(torch.arange(n2))
        pos_emb2 = pos_emb2.to(device)
        x2 = x2 + pos_emb2

        # queries
        n3 = x_embed3.shape[1]
        # [batch, 1900]
        x3 = self.dropout_emb(x_embed3)
        pos_emb3 = max_standardization(torch.arange(n3))
        pos_emb3 = pos_emb3.to(device)
        x3 = x3 + pos_emb3

        # 第三层cross
        atten_state1 = self.input1(x1).view(x1.shape[0], -1, 8)
        atten_state2 = self.input2(x2).view(x2.shape[0], -1, 8)
        atten_state3 = self.input3(x3).view(x3.shape[0], -1, 8)
        atten_state_tower1 = atten_state1
        atten_state_tower2 = atten_state1
        atten_state_tower3 = atten_state2
        atten_state_tower4 = atten_state2
        atten_state_tower5 = atten_state3
        atten_state_tower6 = atten_state3

        for layer in self.cross_atten_tower1:
            atten_state_tower1 = layer(atten_state_tower1, atten_state2)
        for layer in self.cross_atten_tower2:
            atten_state_tower2 = layer(atten_state_tower2, atten_state3)
        for layer in self.cross_atten_tower3:
            atten_state_tower3 = layer(atten_state_tower3, atten_state1)
        for layer in self.cross_atten_tower4:
            atten_state_tower4 = layer(atten_state_tower4, atten_state3)
        for layer in self.cross_atten_tower5:
            atten_state_tower5 = layer(atten_state_tower5, atten_state1)
        for layer in self.cross_atten_tower6:
            atten_state_tower6 = layer(atten_state_tower6, atten_state2)

        intermediate = torch.cat((atten_state_tower1,atten_state_tower2,atten_state_tower3,atten_state_tower4,atten_state_tower5,atten_state_tower6),1)
        if return_intermediate:
            return intermediate  # 返回某一层的特征
            
        score = self.cos(self.dense_out(self.swish(intermediate)))
        score = self.to_score(score)
        return score.squeeze(1)

class multi_layer_attention_no_cross(nn.Module):
    def __init__(
            self,
            *,
            dropout,
            first_self_query_dim=32, first_self_return_dim=512, first_self_num_head=1, first_self_dropout=0.15, first_self_encoder_with_res=True, first_self_residual_coef=1,
            self_deep=1, 
            deep_self_query_dim=16, deep_self_return_dim=128, deep_self_num_head=1, deep_self_dropout=0.15, deep_self_encoder_with_res=True, deep_self_residual_coef=1,
            deep_cross_query_dim=8, deep_cross_return_dim=32, deep_cross_num_head=1, deep_cross_dropout=0.15, deep_cross_encoder_with_res=True, deep_cross_residual_coef=0.5,
            out_scores=2,
            **kwargs
    ):
        super().__init__()
        self.dropout_emb = nn.Dropout(p=dropout)
        self.input1 = nn.LazyLinear(1024)
        self.input2 = nn.LazyLinear(1024)
        self.input3 = nn.LazyLinear(1024)
        self.selfAttentionInput1 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput2 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)
        self.selfAttentionInput3 = SelfAttention(query_dim=first_self_query_dim, return_dim=first_self_return_dim, num_heads=first_self_num_head, dropout=first_self_dropout, encoder_with_res=first_self_encoder_with_res, residual_coef=first_self_residual_coef)

        self_atten_layer1 = []
        self_atten_layer2 = []
        self_atten_layer3 = []
        for i in range(self_deep):
            self_atten_layer1.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer2.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
            self_atten_layer3.append(nn.Sequential(SelfAttention(query_dim=deep_self_query_dim, return_dim = deep_self_return_dim, num_heads=deep_self_num_head, dropout=deep_self_dropout, encoder_with_res=deep_self_encoder_with_res, residual_coef=deep_self_residual_coef)))
        self.self_atten_tower1 = nn.Sequential(*self_atten_layer1)
        self.self_atten_tower2 = nn.Sequential(*self_atten_layer2)
        self.self_atten_tower3 = nn.Sequential(*self_atten_layer3)

        self.cross_atten_tower1 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower2 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower3 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower4 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower5 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.cross_atten_tower6 = nn.ModuleList([CrossAttention(query_dim=deep_cross_query_dim, return_dim=deep_cross_return_dim, num_heads=deep_cross_num_head, dropout=deep_cross_dropout, encoder_with_res=deep_cross_encoder_with_res, residual_coef=deep_cross_residual_coef) for _ in range(1)])
        self.swish = Swish()
        # self.swish = ClippedLinearActivation()
        self.cos = CosineNegActivation()
        self.dense_out = nn.LazyLinear(16)
        self.to_score = nn.Linear(16, out_scores)

    def forward(self, x_embed1, x_embed2, x_embed3, device, first_self_query_dim=32, deep_self=True, deep_self_query_dim=16, deep_cross_query_dim=8, return_intermediate=False):
        n1 = x_embed1.shape[1]
        # [batch, 1900]
        x1 = self.dropout_emb(x_embed1)
        pos_emb1 = max_standardization(torch.arange(n1))
        pos_emb1 = pos_emb1.to(device)
        x1 = x1 + pos_emb1

        # queries
        n2 = x_embed2.shape[1]
        # [batch, 1900]
        x2 = self.dropout_emb(x_embed2)
        pos_emb2 = max_standardization(torch.arange(n2))
        pos_emb2 = pos_emb2.to(device)
        x2 = x2 + pos_emb2

        # queries
        n3 = x_embed3.shape[1]
        # [batch, 1900]
        x3 = self.dropout_emb(x_embed3)
        pos_emb3 = max_standardization(torch.arange(n3))
        pos_emb3 = pos_emb3.to(device)
        x3 = x3 + pos_emb3

        # 第一层self 
        x1 = self.input1(x1).view(x1.shape[0], -1, first_self_query_dim)
        x2 = self.input2(x2).view(x2.shape[0], -1, first_self_query_dim)
        x3 = self.input3(x3).view(x3.shape[0], -1, first_self_query_dim)
        # self_flatten
        atten_state1 = self.selfAttentionInput1(x1).view(x1.shape[0], -1)
        atten_state2 = self.selfAttentionInput2(x2).view(x2.shape[0], -1)
        atten_state3 = self.selfAttentionInput3(x3).view(x3.shape[0], -1)

        # 第二层self
        if deep_self:
            atten_state1 = atten_state1.view(atten_state1.shape[0], -1, deep_self_query_dim)
            atten_state2 = atten_state2.view(atten_state2.shape[0], -1, deep_self_query_dim)
            atten_state3 = atten_state3.view(atten_state3.shape[0], -1, deep_self_query_dim)
            atten_state1 = self.self_atten_tower1(atten_state1)
            atten_state2 = self.self_atten_tower2(atten_state2)
            atten_state3 = self.self_atten_tower3(atten_state3)

        atten_state1 = atten_state1.view(atten_state1.shape[0], -1)
        atten_state2 = atten_state2.view(atten_state2.shape[0], -1)
        atten_state3 = atten_state3.view(atten_state3.shape[0], -1)
        
        intermediate = torch.cat((atten_state1, atten_state2, atten_state3),1)
        if return_intermediate:
            return intermediate  # 返回某一层的特征
            
        score = self.cos(self.dense_out(self.swish(intermediate)))
        score = self.to_score(score)
        return score.squeeze(1)