import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug

torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

EXO = EXOS['AllTheSame']
DEPTH = 2
HEADS = 1
EMBED_SIZE = 9
INNER_SIZE = EMBED_SIZE // HEADS

print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))


# model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
# EXO.test(model, nb_tests=1000)

# set_debug(())
# model(tensor([[0, 0, 0, 2, 2]]))
