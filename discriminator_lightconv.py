import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    DynamicConv,
    FairseqDropout,
    LayerNorm,
    LightweightConv,
    MultiheadAttention,
    PositionalEmbedding,
)