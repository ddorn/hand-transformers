from asyncio import trsock
import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug

torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

EXO = EXOS[0]
DEPTH = 0
HEADS = 1
EMBED_SIZE = 2
INNER_SIZE = EMBED_SIZE // HEADS

print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))

embedding = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [1.0, 0.0]])
unembedding = Tensor([
    [0.0, 0.0, 0.7, 1.0],
    [0.0, 1.0, 0.7, 0.0]]) * 100
pos_encoder = Tensor([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]])

layers = []
model = EXO.mk_model(0, 1, 2, embedding, unembedding, pos_encoder, layers)
EXO.test(model)

# set_debug(())
# model(tensor([[0, 0, 0, 2, 2]]))
