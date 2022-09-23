import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug

torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

EXO = EXOS[1]
DEPTH = 2
HEADS = 1
EMBED_SIZE = 9
INNER_SIZE = EMBED_SIZE // HEADS

print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))

# It is likely impossible with only one attention head

embedding = Tensor([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]])
unembedding = Tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]]) * 100
pos_encoder = Tensor([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])

layer_0_head_0_q = Tensor([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1]]) * 100
layer_0_head_0_k = Tensor([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]])
layer_0_head_0_v = Tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]])
layer_0_head_0 = (layer_0_head_0_q, layer_0_head_0_k, layer_0_head_0_v)

layer_0_head_1_q = Tensor([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]]) * 100
layer_0_head_1_k = layer_0_head_1_q
layer_0_head_1_v = Tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]])
layer_0_head_1 = (layer_0_head_1_q, layer_0_head_1_k, layer_0_head_1_v)

layer_0_heads = [layer_0_head_0, layer_0_head_1]
layer_0_weight = Tensor([
    [1, 0, 0, -1, 0, 0],
    [0, 1, 0, 0, -1, 0],
    [0, 0, 1, 0, 0, -1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]])
layer_0 = (layer_0_heads, layer_0_weight)

layers = [layer_0]

model = EXO.mk_model(1, 2, 3, embedding, unembedding, pos_encoder, layers)
model(torch.tensor([[0, 1, 2]]))
# EXO.test(model, nb_tests=100)