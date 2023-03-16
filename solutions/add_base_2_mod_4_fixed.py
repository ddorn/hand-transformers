import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug
from hand import cat, show_activations, show_model, show_transformer

PB = "AddFixedBase2Mod4"
EXO = EXOS[PB]
DEPTH = 1
HEADS = 1
EMBED_SIZE = 9
INNER_SIZE = EMBED_SIZE // HEADS
MAX_LEN = EXO.tokenizer.max_len
assert MAX_LEN == 5

def pad(x, width=None, height=None):
    """Add zeros on the right and bottom of a 2D tensor so that its size is [bottom, right]."""
    if width is None:
        width = x.shape[1]
    if height is None:
        height = x.shape[0]
    return cat(pad_right(x, width), torch.zeros(height - x.shape[0], width))

def pad_right(x, pad=EMBED_SIZE):
    """Add zeros on the right of a 2D tensor so that its width is [pad]."""
    return cat([x, torch.zeros(x.shape[0], pad - x.shape[1])])

def add_sol():
    Z = torch.zeros
    embedding = pad_right(Tensor([
        [0],  # pad (not used)
        [0],  # 0
        [1],  # 1
    ]))
    unembedding = embedding.T

    pos_encoding = pad_right(Tensor([
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],  # TODO: find a good value for this one
    ]))

    # Query: tokens of the same number (a=0 or b=1)
    wq_00 = pad(Tensor([
        [1, 1],  # token embedding
        [1, 0],  # pos: in A
        [0, 1],  # pos: in B
    ]), EMBED_SIZE, INNER_SIZE)
    # Key: id of the number (0 or 1)
    wk_00 = pad(Tensor([
        [0, 0],  # token embedding
        [1, 0],  # pos: in A
        [0, 1],  # pos: in B
    ]), EMBED_SIZE, INNER_SIZE) * 10
    # Value: 2**value of the token
    wv_00 = pad(Tensor([
        [0],  # token
        [0],  # pos: in A
        [0],  # pos: in B
        [2],  # pos: is leading digit
        [1],  # pos: is last digit
    ]), EMBED_SIZE, INNER_SIZE) * 2  # *2 because attention averages the values of the 2 digits (and doesn't sum them)

    """
    # Query: the matching position in the other number (ie. flip bits in pos 2 & 3)
    wq_00 = pad(Tensor([  # shape: EMBED_SIZE x INNER_SIZE
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]), height=EMBED_SIZE, width=INNER_SIZE)
    # Key: The token's position
    wk_00 = pad(Tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]), height=EMBED_SIZE, width=INNER_SIZE)
    # Value: the digit
    v_00 = pad(Tensor([[1]]), height=EMBED_SIZE, width=INNER_SIZE)
    """

    head_00 = (wq_00, wk_00, wv_00)

    # we copy the attended token the 6-th column of the residual stream
    weight_0 = Z(EMBED_SIZE, EMBED_SIZE)
    weight_0[0, 5] = 1

    layers = [
        ([head_00], weight_0),
    ]

    model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE,
        embedding, unembedding, pos_encoding, layers)

    return model




if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

    # print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))

    model = add_sol()
    import matplotlib.pyplot as plt
    show_transformer(model)
    show_activations(model, EXO.tokenizer.encode(["01001"]))
    plt.show()

    # model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
    # EXO.test(model, nb_tests=1000)

    # set_debug(())
    # model(tensor([[0, 0, 0, 2, 2]]))
