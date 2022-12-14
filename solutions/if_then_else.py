from typing import List
import torch
from torch import Tensor, tensor
from exos import EXOS, set_debug
from hand import cat, pad, Transformer, MultiAttentionHead, ResidualMLP, attend_abs_position, copy

PB = "IfThenElse"
EXO = EXOS[PB]
DEPTH = 1
HEADS = 2
EMBED_SIZE = 12
INNER_SIZE = EMBED_SIZE // HEADS
VOCAB = EXO.vocab_size
TOKENS = EXO.tokenizer.max_len
HEAD_SHAPE = (EMBED_SIZE, INNER_SIZE)


def if_then_else():
    O = torch.zeros
    p = torch.nn.Parameter

    embedding = pad(torch.eye(VOCAB - 1), EMBED_SIZE, VOCAB, 0, 1)
    pos_encoding = pad(torch.eye(TOKENS), EMBED_SIZE, TOKENS, VOCAB - 1, 0)

    model = Transformer(VOCAB, EMBED_SIZE, DEPTH, HEADS, pos_encoding, (4, ))
    model.embedding.weight = p(embedding)

    # Usage of residual stream dimensions
    TOK = [0, 1]
    POS = [2, 3, 4]

    heads: MultiAttentionHead = model.blocks[0].heads

    # Head 1: copy the condition token
    k, q = attend_abs_position(dims=POS, permutation=[0, 0, 0], shape=HEAD_SHAPE)
    v = p(copy(in_dims=TOK, out_dims=[0, 1], shape=HEAD_SHAPE))
    heads[0].set(k, q, v)

    # Head 2: copy the other option token (token 1 or 2 when pos is 2 or 1 resp.)
    k, q = attend_abs_position(dims=POS, permutation=[0, 2, 1], shape=HEAD_SHAPE)
    v = copy(in_dims=TOK, out_dims=[0, 1], shape=HEAD_SHAPE)
    heads[1].set(k, q, v)

    # Weight: copy the outputs of the two heads (2 dims each) at the end of the residual stream
    model.blocks[0].weight = p(
        copy(in_dims=[0, 1, 6, 7],
             out_dims=[5, 6, 7, 8],
             shape=(EMBED_SIZE, EMBED_SIZE)))

    mlp: ResidualMLP = model.blocks[1]
    # Computes [curent_token - 10·t_0, other_token + 10·t_0] for i in [1, 2]
    s = 10.0  # To separate the two options, the good one is sent to positive values
    mlp.layers[0].weight = p(torch.tensor([  # Size: 4 * EMBED_SIZE
        [1, 0, 0, 0, 0, s, -s, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, s, -s, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -s, s, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -s, s, 0, 1, 0, 0, 0],
    ]))
    mlp.layers[0].bias = p(O(4))
    t = 1.0  # To game the softmax
    mlp.layers[1].weight = p(pad([[t, 0, t, 0], [0, t, 0, t]], 4, EMBED_SIZE, 0, EMBED_SIZE - 2))
    mlp.layers[1].bias = p(torch.tensor([0] * (EMBED_SIZE - 2) + [-s*t, -s*t]))

    model.unembedding = p(copy(in_dims=[EMBED_SIZE - 2, EMBED_SIZE - 1],
                               out_dims=[1, 2],
                               shape=(EMBED_SIZE, VOCAB)) * 10)  # To game the softmax

    return model


if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)

    print(EXO.print_template(DEPTH, HEADS, INNER_SIZE, default="0"))

    # model = EXO.mk_model(DEPTH, HEADS, INNER_SIZE, embedding, unembedding, pos_encoder, layers)
    # EXO.test(model, nb_tests=1000)

    # set_debug(())
    # model(tensor([[0, 0, 0, 2, 2]]))
