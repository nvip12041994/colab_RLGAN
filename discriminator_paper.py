import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax,
    DynamicConv,
    FairseqDropout,
    LayerNorm,
    LightweightConv,
    MultiheadAttention,
    PositionalEmbedding,
)


def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            # param.data.uniform_(-0.1, 0.1)
            nn.init.kaiming_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Discriminator_lightconv(nn.Module):
    
    def __init__(self, args, task, kernel_size=0):
        super(Discriminator_lightconv, self).__init__()
        self.embed_dim = args.model.encoder_embed_dim
        self.conv_dim = args.model.encoder_embed_dim #args.model.encoder_conv_dim 512
        self.dropout_module = FairseqDropout(
            0.1, module_name=self.__class__.__name__
        )
        
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )
        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        embed_tokens = build_embedding(
                src_dict, args.model.encoder_embed_dim, args.model.encoder_embed_path
            )
        
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.model.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                args.model.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.model.encoder_learned_pos,
            )
            if not args.model.no_token_positional_embeddings
            else None
        )
        
        self.conv_thought_space = nn.Sequential(
            Conv1d(in_channels=512,
                   out_channels=128,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
       
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            Linear(3456, 20),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(20, 20),
            nn.ReLU(),
            Linear(20, 1),
        )
        
    def forward(self, src_tokens,tgt_tokens):
        def fixed_padding(a,b,num_pad):
            a_pad = F.pad(a, (0, num_pad-a.shape[2]))
            b_pad = F.pad(a, (0, num_pad-b.shape[2]))
            return a_pad, b_pad

        def pad_to_same_size(a,b):
            if a.shape[2]%2:
                a = F.pad(a, (1,0), "constant", 0)
            if b.shape[2]%2:
                b = F.pad(b, (1,0), "constant", 0)

            if a.shape[2] > b.shape[2]:
                num_pad = a.shape[2] - b.shape[2]
                padding = nn.ConstantPad1d(num_pad//2,0)
                b_pad = padding(b)
                a_pad = a    
            elif a.shape[2] < b.shape[2]:
                num_pad = b.shape[2] - a.shape[2]
                padding = nn.ConstantPad1d(num_pad//2,0)
                a_pad = padding(a)
                b_pad = b    
            else:
                a_pad = a
                b_pad = b
            return a_pad, b_pad
        def stack_matrix_interleave(a,b,dim_stack):
            size = list(a.shape)
            size[dim_stack]=2
            size.insert(0,a.shape[dim_stack])
            size = tuple(size)
            out = torch.empty(size)
            
            for i in range(a.shape[dim_stack]):
                stack = []
                stack.append(a[i,:,:])
                stack.append(b[i,:,:])        
                tmp = torch.stack(stack,dim=dim_stack)        
                out[i] = tmp
            return out
        batch_size = src_tokens.shape[0]
        
        # embed tokens and positions
        src = self.embed_scale * self.embed_tokens(src_tokens)
        tgt = self.embed_scale * self.embed_tokens(tgt_tokens)
        
        if self.embed_positions is not None:
            src += self.embed_positions(src_tokens)
            tgt += self.embed_positions(tgt_tokens)
        src = self.dropout_module(src)
        tgt = self.dropout_module(tgt)

        # B x T x C -> T x B x C
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        src = src.permute(1,2,0)
        tgt = tgt.permute(1,2,0)
        
        out_src = self.conv_thought_space(src)
        out = out.contiguous().view(batch_size, -1)
        out = torch.sigmoid(self.classifier(out))
        
        return out
