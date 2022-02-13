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
glu=True
#conv_type="lightweight"
conv_type="dynamic"


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
        # Config how many layer and attention head for lightconv
        encoder_layers = 7
        encoder_kernel_size_list = [3,7,15,31,31,31,31]
    
        dynamic_encoder_config = Encoder_config(encoder_conv_dim = 512,
                                            encoder_embed_dim = embed_dim,
                                            encoder_glu = True,
                                            encoder_conv_type = "dynamic",
                                            weight_softmax = True,
                                            encoder_attention_heads = 4,
                                            encoder_normalize_before = False,
                                            encoder_ffn_embed_dim = 1024,
                                            weight_dropout = 0.1,
                                            dropout = 0.1,
                                            relu_dropout = 0.0,
                                            input_dropout= 0.0)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                LightConvEncoderLayer(
                    dynamic_encoder_config, kernel_size=encoder_kernel_size_list[i]
                )
                for i in range(encoder_layers)
            ]
        )
        
        self.conv_thought_space = nn.Sequential(
            Conv2d(in_channels=2,
                   out_channels=128,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=128,
                   out_channels=64,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=64,
                   out_channels=32,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32,
                   out_channels=16,
                   kernel_size=3,
                   stride=1,
                   padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
       
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            Linear(6144, 20),
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

        # encoder layers
        for layer in self.layers:
            src = layer(src, encoder_padding_mask)
            tgt = layer(tgt, encoder_padding_mask)
        #torch.Size([56, 3, 512])

        src = src.permute(1,2,0)
        tgt = tgt.permute(1,2,0)
        src,tgt = pad_to_same_size(src,tgt)
        src,tgt = fixed_padding(src,tgt,200)
        stack_matrix = stack_matrix_interleave(src,tgt,dim_stack=0)
        stack_matrix = stack_matrix.to(device='cuda')
        out = self.conv_thought_space(stack_matrix)
        out = out.contiguous().view(batch_size, -1)
        out = torch.sigmoid(self.classifier(out))
        del src
        del tgt
        del stack_matrix
        torch.cuda.empty_cache()
        return out


class Encoder_config:
    def __init__(self,encoder_conv_dim,encoder_embed_dim,encoder_glu,encoder_conv_type,weight_softmax,
                 encoder_attention_heads,encoder_normalize_before,encoder_ffn_embed_dim,
                 weight_dropout,dropout,relu_dropout,input_dropout):
        self.encoder_conv_dim = encoder_conv_dim
        self.encoder_glu = encoder_glu
        self.encoder_conv_type = encoder_conv_type
        self.weight_softmax = weight_softmax
        self.encoder_normalize_before = encoder_normalize_before
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.weight_dropout = weight_dropout
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_embed_dim = encoder_embed_dim
        
        

class LightConvEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        kernel_size: kernel size of the convolution
    """

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.conv_dim = args.encoder_conv_dim
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        if args.encoder_glu:
            self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = Linear(self.embed_dim, self.conv_dim)
            self.act = None
        if args.encoder_conv_type == "lightweight":
            self.conv = LightweightConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout,
            )
        elif args.encoder_conv_type == "dynamic":
            self.conv = DynamicConv(
                self.conv_dim,
                kernel_size,
                padding_l=padding_l,
                weight_softmax=args.weight_softmax,
                num_heads=args.encoder_attention_heads,
                weight_dropout=args.weight_dropout,
            )
        else:
            raise NotImplementedError
        self.linear2 = Linear(self.conv_dim, self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.relu_dropout_module = FairseqDropout(
            args.relu_dropout, module_name=self.__class__.__name__
        )
        self.input_dropout_module = FairseqDropout(
            args.input_dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_module(x)
        x = self.linear1(x)
        if self.act is not None:
            x = self.act(x)
        # if encoder_padding_mask is not None:
        #     x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv(x)
        x = self.linear2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = self.relu_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def extra_repr(self):
        return (
            "dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}".format(
                self.dropout_module.p,
                self.relu_dropout_module.p,
                self.input_dropout_module.p,
                self.normalize_before,
            )
        )