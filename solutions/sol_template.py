import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug
from hand import cat

PB = "Sort"
EXO = EXOS[PB]
DEPTH = 2
HEADS = 1
EMBED_SIZE = 9
INNER_SIZE = EMBED_SIZE // HEADS


def sol():

    embedding = ...
    unembedding = embedding.T
    pos_encoding = ...

    wq_00 = ...
    wk_00 = ...
    wv_00 = ...
    head_00 = (wq_00, wk_00, wv_00)
    weight_0 = ...

    layers = [
        ([head_00], weight_0),
    ]

    model = EXO.mk_model(DEPTH, HEADS, EMBED_SIZE, INNER_SIZE,
        embedding, unembedding, pos_encoding, layers)

    return model

if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

    print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))


    # model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
    # EXO.test(model, nb_tests=1000)

    # set_debug(())
    # model(tensor([[0, 0, 0, 2, 2]]))
