import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug

PB = "Sort"
EXO = EXOS[PB]
DEPTH = 2
HEADS = 1
EMBED_SIZE = 7
INNER_SIZE = EMBED_SIZE // HEADS

def sort():
    embedding = Tensor([
        [0, 0, 0, 0, 0, 0, 0],  # pad token
        [1, 0, 0, 0, 0, 0, 0],  # 0
        [0, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 1, 0, 0, 0, 0],  # 2
        [0, 0, 0, 1, 0, 0, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0],  # 4
        [0, 0, 0, 0, 0, 1, 0],  # 5
        [0, 0, 0, 0, 0, 0, 1],  # |
    ])
    unembedding = embedding.T



if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

    print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))



    # model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
    # EXO.test(model, nb_tests=1000)

    # set_debug(())
    # model(tensor([[0, 0, 0, 2, 2]]))