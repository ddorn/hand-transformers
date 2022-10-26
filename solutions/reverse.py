from asyncio import trsock
import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug
from hand import cat

torch.set_printoptions(precision=1, sci_mode=False, linewidth=200)


I = torch.eye
O = torch.zeros
F = torch.full

EXO = EXOS['Reverse']
DEPTH = 2
HEADS = 2
EMBED_SIZE = 4 + 8 + 8  # tokens + pos + working space
INNER_SIZE = EMBED_SIZE // HEADS

# print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))

embedding = cat(O(1, EMBED_SIZE),
                [I(4), O(4, EMBED_SIZE - 4)])
unembedding = embedding.T
pos_encoder = torch.cat((O(8, 4), I(8), O(8, 8)), dim=1)

# This head finds the position of the | character and outputs it in the 8 first dims
layer_0_head_0_q = cat(
    [F((4, 4), 1.0), O(4, INNER_SIZE - 4)],
    O(EMBED_SIZE - 4, INNER_SIZE)
)
layer_0_head_0_k = O(EMBED_SIZE, INNER_SIZE)
layer_0_head_0_k[3, 0] = 100  # Find the |, which is embedded as [0, 0, 0, 1]
layer_0_head_0_v = cat(
    O(4, INNER_SIZE),
    [I(8), O(8, INNER_SIZE - 8)],
    O(8, INNER_SIZE),
)
layer_0_head_0 = (layer_0_head_0_q, layer_0_head_0_k, layer_0_head_0_v)

# This head just outputs the current position in it last 8 dims
layer_0_head_1_q = cat(O(4, INNER_SIZE), [I(8), O(8, 2)], O(8, INNER_SIZE))
layer_0_head_1_k = layer_0_head_1_q * 100
layer_0_head_1_v = layer_0_head_1_q
layer_0_head_1 = (layer_0_head_1_q, layer_0_head_1_k, layer_0_head_1_v)


layer_0_heads = [layer_0_head_0, layer_0_head_1]
# Computes p(|) - d(|, current) by comparing the outputs
# of the two heads.
# Outputs a vector in the working space whose highest component
# is the position of the token to replicate.
layer_0_weight = cat(
    [O(8, EMBED_SIZE - 8), tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])],
    O(4, EMBED_SIZE),  # last two cols of first head and first 2 of second head are not used.
    [O(8, EMBED_SIZE - 8), tensor([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 2, 0, 2, 0, 2, 0],
    ])],
)
layer_0 = (layer_0_heads, layer_0_weight)

layer_1_head_0 = (
    cat(  # Keys
        O(4, INNER_SIZE),
        O(8, INNER_SIZE),
        [O(8, 2), I(8)],
    ),
    cat(  # Queries
        O(4, INNER_SIZE),
        [O(8, 2), I(8)],
        O(8, INNER_SIZE),
    ) * 100,
    cat(  # Values
        [I(4), O(4, INNER_SIZE - 4)],
        O(8 + 8, INNER_SIZE),
    )
)
layer_1_head_1 = (O(EMBED_SIZE, INNER_SIZE),) * 3

layer_1_heads = [layer_1_head_0, layer_1_head_1]
layer_1_weight = I(EMBED_SIZE) * 100
layer_1 = (layer_1_heads, layer_1_weight)

layers = [layer_0, layer_1]


model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
EXO.test(model, nb_tests=100)

# set_debug((..., 'layer'), (1,), ("embed+pos",))
# model(tensor([[0, 0, 0, 0, 2, 1,  4, 1]]))

# print(layer_0_weight)