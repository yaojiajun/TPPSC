import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
import taichi as ti

import einops
from einops import rearrange

ti.init()



class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)  # skip connection

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #100*128
        pe = pe.unsqueeze(0) #1*100*128
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class PositionalDecoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalDecoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x - self.pe[:,:x.size(1),:]
        return self.dropout(x)


class MultiHeadPosCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadPosCompat, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))


        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, pos, st_edge):
       # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = pos.size()
        posflat = pos.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(posflat, self.W_query).view(shp)
        K = torch.matmul(posflat, self.W_key).view(shp)

        Q.shape
        att=torch.matmul(Q, K.transpose(2, 3))
        att.shape

        st_edge.shape
        compatibility=(att+st_edge)*self.norm_factor
        # att = []
        # for i in range(graph_size):
        #     aa = Q[:, :, i, :][:, :, None, :]
        #     aa2 = (edg[:, :, i][:, :, None]).expand(batch_size, graph_size, graph_size)
        #     aa2.shape
        #     bb = torch.matmul(aa2, K)
        #     bb.shape
        #     a = torch.matmul(aa, bb.transpose(2, 3))
        #     att.append(a)
        # compatibility = torch.stack(att, 0).squeeze(3).permute(1, 2, 0, 3)
        # compatibility.shape  # 8,2,51,51


        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        return compatibility


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.init_embed_depot = nn.Linear(2, embed_dim)

        self.init_embed = nn.Linear(9, embed_dim)  # node_embedding

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, q, p, h=None, mask=None):

        if h is None:
            h = q  # compute self-attention



        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()  # input_dim=embed_dim
        n_query = q.size(1)  # =graph_size+1
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        hflat.shape
        qflat.shape
        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        self.W_query

        # Calculate queries, (n_heads, batch_size, n_query, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf
        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)  # [n_heads, batrch_size, n_query, val_size]

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),  # [batch_size, n_query, n_heads*val_size]
            self.W_out.view(-1, self.embed_dim)  # [n_head*key_dim, embed_dim]
        ).view(batch_size, n_query, self.embed_dim)

        return out


class MultiHeadAttentionNew(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionNew, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, out_source_attn, st_edge, h=None,): #d_q, out_source_attn,
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = q.size()
        # edg=edg
        hflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (8, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        # edg=edg.view(shp)
        # Calculate compatibility (n_heads, bastch_size, n_query, graph_size)

        compatibility=(torch.matmul(Q, K.transpose(2, 3))+st_edge)*self.norm_factor
        compatibility = torch.cat((compatibility, out_source_attn), 0)

        attn_raw = compatibility.permute(1, 2, 3, 0)
        attn = self.score_aggr(attn_raw).permute(3, 0, 1, 2)

        heads = torch.matmul(F.softmax(attn, dim=-1), V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out, out_source_attn , st_edge


class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2, input3):
        out1, out2, out3= self.MHA_sublayer(input1, input2, input3)
        return self.FFandNorm_sublayer(out1), out2, out3

class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttentionNew(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input1, input2, input3):
        # Attention and Residual connection
        out1, out2, out3= self.MHA(input1, input2, input3)
        # Normalization
        return self.Norm(out1 + input1), out2, out3


class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim, bias=False)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias=False)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class MultiHeadAttentionNewDyn(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            elem_dim=20,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionNewDyn, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.elem_dim = elem_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, out_source_attn, st_edge, h=None,): #d_q, out_source_attn,
        # h should be (batch_size, graph_size, input_dim)
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, time, graph_size, input_dim)
        #if len(h.size()) == 4:
        time, batch_size, graph_size, input_dim = h.size()
        # h should be (batch_size, graph_size, input_dim)
        #else:
        #  batch_size, graph_size, input_dim = h.size()

        n_time = q.size(0)
        assert q.size(1) == batch_size
        assert q.size(3) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(batch_size*time*graph_size, input_dim)
        qflat = q.contiguous().view(batch_size*time*graph_size, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, time, batch_size, graph_size,-1)
        shp_q = (self.n_heads, time, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        att=torch.matmul(Q, K.transpose(3, 4))

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * (att + st_edge)

        out_source_attn1=out_source_attn[:,None,:,:,:].expand(-1,time,-1,-1,-1)

        compatibility = torch.cat((compatibility, out_source_attn1), 0)
        attn_raw = compatibility.permute(1, 2, 3, 4, 0)
        attn = self.score_aggr(attn_raw).permute(4, 0, 1, 2, 3)

        # Optionally apply mask to prevent attention
        # if mask is not None:
        #     mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
        #     compatibility[mask] = -np.inf

        attn = torch.softmax(attn, dim=-1)
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(  n_time, batch_size, graph_size, self.embed_dim)

        return out, out_source_attn , st_edge

class MultiHeadEncoderDyn(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoderDyn, self).__init__()

        self.MHA_sublayer_dyn = MultiHeadAttentionsubLayerDyn(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer_dyn = FFandNormsubLayerDyn(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2, input3):
        out1, out2, out3= self.MHA_sublayer_dyn(input1, input2, input3)
        return self.FFandNorm_sublayer_dyn(out1),out2, out3


class MultiHeadAttentionsubLayerDyn(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayerDyn, self).__init__()

        self.MHA_dyn = MultiHeadAttentionNewDyn(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input1, input2, input3):
        # Attention and Residual connection
        out1, out2 , out3 = self.MHA_dyn(input1, input2, input3) #, out2
        # Normalization
        return self.Norm(out1 + input1), out2, out3

class FFandNormsubLayerDyn(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayerDyn, self).__init__()

        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim, bias=False)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias=False)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)



class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            print('stdv', stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())  # [batch_size, graph_size+1, embed_dim]
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


# Encoder part, hi_hat and hi^l
class MultiHeadAttentionLayer(nn.Sequential):
# multihead attention -> skip connection, normalization -> feed forward -> skip connection, normalization
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.positional_encode = PositionalEncoding(d_model=embed_dim, dropout=0.0)

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, y, z, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes   h/x: [batch_size, graph_size+1, embed_dim]
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h, y, z)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
