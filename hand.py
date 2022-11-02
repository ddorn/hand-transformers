from contextlib import contextmanager
from distutils.log import debug
import enum
import math
from textwrap import dedent
from typing import Callable, List, Optional, Tuple, Type, Union

import einops
import matplotlib.pyplot as plt
import torch
from torchtyping import TensorType, patch_typeguard
import tqdm
from typeguard import typechecked

patch_typeguard()

# ------------------------- #
# Reimplementation of GPT-2 #
# ------------------------- #

class CharTokenizer:

    def __init__(self, vocab: str, max_len: int):
        """A tokenizer that just maps characters to indices.

        Args:
            vocab (str): A string containing all the characters that can be used.
            max_len (int): The maximum length of a prompt.

        Example:
            >>> tokenizer = DumbTokenizer("abc", 3)
            >>> tokenizer.encode(["ab", "c"])
            tensor([[0, 1, 2],
                    [0, 0, 3]])
            >>> tokenizer.decode(torch.tensor([[0, 1, 2], [0, 0, 3]]))
            ['ab', 'c']
        """

        self.vocab = [""] + list(vocab)
        self.max_len = max_len
        self.vocab_size = len(self.vocab)

    @typechecked
    def encode(self,
               prompts: List[str],
               pad: bool = True) -> TensorType['batch', 'max_len']:
        if any(len(prompt) > self.max_len for prompt in prompts):
            print("Warning: prompts longer than max_len will be truncated.")

        prompts = [prompt[-self.max_len:] for prompt in prompts]

        return torch.tensor([[0] * pad * (self.max_len - len(prompt)) +
                             [self.vocab.index(char) for char in prompt]
                             for prompt in prompts])

    @typechecked
    def decode(self, tokens: TensorType['batch', 'token']) -> List[str]:
        return [
            "".join(self.vocab[token] for token in prompt) for prompt in tokens
        ]


class PositionEncoder(torch.nn.Module):  # Not used
    """A positional encoder that adds (co)sinusoidal frequencies to the input."""

    @typechecked
    def forward(
        self, x: TensorType['batch', 'tokens', 'embedding_size']
    ) -> TensorType['batch', 'tokens', 'embedding_size']:
        tokens, embedding_size = x.shape[1], x.shape[2]
        assert embedding_size % 2 == 0, "embedding size must be even."

        pe = torch.zeros(tokens, embedding_size)
        positions = torch.arange(0, tokens)
        scales = (1 / 10000)**(torch.arange(0, embedding_size // 2) /
                               embedding_size)
        coeffs = torch.einsum("i,j->ij", positions, scales)
        sins = torch.sin(coeffs)
        coss = torch.cos(coeffs)
        pos_encoding = einops.rearrange(
            torch.stack([sins, coss]), "type tokens emb -> tokens (emb type)")
        return x + pos_encoding


class AttentionHead(torch.nn.Module):

    def __init__(self, embedding_size: int, out_size: int, layer:int=None, n:int=None) -> None:
        super().__init__()

        self.layer = layer
        self.n = n

        self.queries = torch.nn.Parameter(torch.randn(embedding_size, out_size))
        self.keys = torch.nn.Parameter(torch.randn(embedding_size, out_size))
        self.values = torch.nn.Parameter(torch.randn(embedding_size, out_size))

    @typechecked
    def forward(
        self, x: TensorType['batch', 'tokens', 'embedding_size']
    ) -> TensorType['batch', 'tokens', 'out_size']:

        q = torch.einsum("bte,ei->bti", x, self.queries)
        k = torch.einsum("bte,ei->bti", x, self.keys)
        v = torch.einsum("bte,ei->bti", x, self.values)

        debug(q, self.layer, self.n, 'q')
        debug(k, self.layer, self.n, 'k')
        debug(v, self.layer, self.n, 'v')

        qk = torch.einsum("bti,bTi->btT", q, k)
        s = torch.softmax(qk / q.shape[-1]**0.5, dim=-1)
        out = torch.einsum("btT,bTi->bti", s, v)

        debug(qk, self.layer, self.n, 'fit')
        debug(s, self.layer, self.n, 'attention')
        debug(out, self.layer, self.n, 'head')

        return out


class MultiAttentionHead(torch.nn.Module):

    def __init__(self, embedding_size: int, heads: int, layer:int=None) -> None:
        assert embedding_size % heads == 0, "embedding size must be divisible by heads."
        super().__init__()
        self.layer = layer

        out_size = embedding_size // heads

        self.heads = torch.nn.ModuleList(
            [AttentionHead(embedding_size, out_size, layer=layer, n=n) for n in range(heads)])
        self.weight = torch.nn.Parameter(torch.randn(embedding_size, embedding_size))

    def forward(self, x: TensorType['b', 't',
                                    'emb']) -> TensorType['b', 't', 'out']:
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        debug(combined, self.layer, "heads-combined")
        multihead = combined @ self.weight
        debug(self.weight, self.layer, "heads-weight")
        debug(multihead, self.layer, "multihead")
        output = multihead + x
        debug(output, self.layer, "layer")
        return output


class ResidualMLP(torch.nn.Module):
    def __init__(self, embeding_size: int, *layers_dims: int, layer:int=None) -> None:
        super().__init__()
        dims = embeding_size, *layers_dims, embeding_size
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

    def forward(self, x: TensorType['batch', 'tokens', 'embedding_size']) -> TensorType['batch', 'tokens', 'embedding_size']:
        initial = x
        for l, layer in enumerate(self.layers):
            x = layer(x)
            debug(x, self.layer, "mlp", l)
            x = torch.relu(x)
            debug(x, self.layer, "mlp", l, "relu")

        x = x + initial
        debug(x, self.layer, "mlp", "residual")
        return x

class Transformer(torch.nn.Module):

    def __init__(self, voc_size: int, embedding_size: int, depth: int,
                 heads: int, pos_encoder: TensorType['max_prompt_len', 'embedding_size'],
                 mlp_dims: Optional[Tuple[int, ...]]=None) -> None:
        super().__init__()

        self.depth = depth
        self.heads = heads
        self.voc_size = voc_size
        self.embedding = torch.nn.Embedding(voc_size, embedding_size)
        # self.position_encoder = PositionEncoder()
        self.position_encoder = pos_encoder

        heads = [
            MultiAttentionHead(embedding_size, heads, layer=layer)
            for layer in range(depth)
        ]
        if mlp_dims is not None:
            mlps = [ResidualMLP(embedding_size, *mlp_dims) for _ in range(depth)]
            # Interleave heads and mlps
            blocks = [block for layer in zip(heads, mlps) for block in layer]
        else:
            blocks = heads

        self.blocks = torch.nn.ModuleList(blocks)
        self.unembedding = torch.nn.Parameter(torch.rand(embedding_size, voc_size))

    def forward(self, x: TensorType['batch', 'token']) -> List[str]:
        embeded = self.embedding(x)
        debug(embeded, "embeded")
        # with_pos = self.position_encoder(embeded)
        with_pos = embeded + self.position_encoder
        debug(with_pos, "embed+pos")
        x = self.blocks(with_pos)
        out = x[:, -1, :].squeeze(1)  # only the last token is the prediction
        unembeded = out @ self.unembedding
        debug(unembeded, "unembeded")
        probas = torch.softmax(unembeded, dim=-1)
        debug(probas, "probas")
        return probas


# ----------------------- #
# Framework for exercises #
# ----------------------- #

class Exercise:

    def __init__(self, name: str, tokenizer: CharTokenizer,
                 generator: Callable[[], Tuple[str, str]]) -> None:
        self.tokenizer = tokenizer
        self.generator = generator
        self.description = generator.__doc__
        self.name = name

    def __repr__(self) -> str:
        examples = [self.generator() for _ in range(5)]
        max_len = max(len(example[0]) for example in examples)
        examples = [
            f"  {example[0]:>{max_len}} → {example[1]}"
            for example in examples
        ]
        examples = '\n'.join(examples)

        voc = [
            f"{idx}: {char if char.strip() else repr(char)}"
            for idx, char in enumerate(self.tokenizer.vocab)
        ]
        voc = '  '.join(voc)

        return f"""{self.name}
{self.description}

Alphabet: {voc}
Input length: {self.tokenizer.max_len}
Examples:
{examples}"""

    @property
    def voc(self) -> List[str]:
        return self.tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def generate(self, n: int) -> Tuple[List[str], List[str]]:
        """Generate n inputs and expected outputs."""
        xs, ys = zip(*[self.generator() for _ in range(n)])
        return list(xs), list(ys)

    @typechecked
    def mk_model(
        self,
        depth: int,
        heads: int,
        inner_size: int,
        embedding: TensorType['voc', 'embedding_size'],
        unembedding: TensorType['embedding_size', 'voc'],
        position_encoder: TensorType['max_len', 'embedding_size'],
        layers: List[Tuple[List[Tuple[TensorType['token', 'inner'],  # Q
                                      TensorType['token', 'inner'],  # K
                                      TensorType['token', 'inner'],  # V
                                      ],  # per head
                                ], TensorType['embedding',
                                              'embedding'],  # W per layer
                           # Feed forward ? norm ?
                           ]],
    ) -> Transformer:
        """Create a model with the given parameters. MLP not supported."""

        # This method was supposed to make it easier to build models
        # for the hackaton, but I don't think it useful for my tinkering.

        # Shorthand
        p = lambda x: torch.nn.Parameter(x, requires_grad=False)

        model = Transformer(self.tokenizer.vocab_size,
                                         inner_size * heads, depth, heads, position_encoder)
        model.embedding.weight = p(embedding)
        model.unembedding = p(unembedding)

        for i, (heads, weight) in enumerate(layers):
            for j, (q, k, v) in enumerate(heads):
                model.blocks[i].heads[j].queries = p(q)
                model.blocks[i].heads[j].keys = p(k)
                model.blocks[i].heads[j].values = p(v)
            model.blocks[i].weight = p(weight)

        return model

    @typechecked
    def test(self,
             model: torch.nn.Module,
             nb_tests: int = 100) -> Tuple[float, float]:

        xs, ys = self.generate(nb_tests)
        xs_enc = self.tokenizer.encode(xs)
        ys_enc = self.tokenizer.encode(ys, pad=False)[:, -1]

        # Count successes and compute loss
        pred_enc = model(xs_enc)
        loss = torch.nn.functional.cross_entropy(pred_enc, ys_enc).item()
        correct = (pred_enc.argmax(dim=-1) == ys_enc).sum().item()

        # Show first 10 predictions
        pred = self.tokenizer.decode(pred_enc.argmax(dim=-1).unsqueeze(-1))
        xs_width = max(len(x) for x in xs)
        for i in range(10):
            expected = "EXPECTED " + ys[i] if ys[i] != pred[i] else "        "
            token_probs = [
                f"{char if char.strip() else repr(char)}: {proba:.2f}"
                for char, proba in zip(self.voc, pred_enc[i].detach().numpy())
            ]
            token_probs = '  '.join(token_probs)
            print(f"{xs[i]:>{xs_width}} → {pred[i]} {expected}\t{token_probs}")
        print(f"Loss: {loss:.2f}  Accuracy: {correct} / {nb_tests}")

        return correct, loss

    def print_template(self,
                       depth: int,
                       heads: int,
                       inner_size: int,
                       default: str = "0.0") -> None:
        voc_size = self.vocab_size
        emb = inner_size * heads

        def mk_matrix(h: int, w: int) -> str:
            nl = "\n" + 12 * " "
            return f"Tensor([{nl}[" + f"],{nl}[".join(
                ", ".join(default for _ in range(w)) for _ in range(h)) + "]])"


        template = f"""
        embedding = {mk_matrix(voc_size, emb)}
        unembedding = {mk_matrix(emb, voc_size)}
        pos_encoder = {mk_matrix(self.tokenizer.max_len, emb)}
        """
        for d in range(depth):
            for h in range(heads):
                template += f"""
        layer_{d}_head_{h}_q = {mk_matrix(emb, inner_size)}
        layer_{d}_head_{h}_k = {mk_matrix(emb, inner_size)}
        layer_{d}_head_{h}_v = {mk_matrix(emb, inner_size)}
        layer_{d}_head_{h} = (layer_{d}_head_{h}_q, layer_{d}_head_{h}_k, layer_{d}_head_{h}_v)
        """
            template += f"""
        layer_{d}_heads = [""" + ", ".join(f"layer_{d}_head_{h}" for h in range(heads)) + f"""]
        layer_{d}_weight = {mk_matrix(emb, emb)}
        layer_{d} = (layer_{d}_heads, layer_{d}_weight)
        """
        template += f"""
        layers = [""" + ", ".join(f"layer_{d}" for d in range(depth)) + "]"

        template = dedent(template)

        return template


def mkexo(name: str, voc: str, input_len: int) -> Exercise:

    def decorator(f: Callable[[], Tuple[str, str]]) -> Exercise:
        return Exercise(name, CharTokenizer(voc, input_len), f)

    return decorator

# ------------------------------ #
# Help for visualizing the model #
# ------------------------------ #

DEBUG = set(("",))  # debug everythin
EllipsisType = type(...)
DEBUG_CALLBACK = None

def debug(value: TensorType, *name: Union[str, int]) -> None:
    for pattern in DEBUG:
        # print('eval', pattern, name)
        for part, pat in zip(name, pattern):
            if pat is not ... and part != pat:
                break  # not a match, next pattern
        else:
            # We have found a pattern that matches
            if DEBUG_CALLBACK is not None:
                DEBUG_CALLBACK(value, *name)
            else:
                print(*name)
                pprint_2d_tensor(value)
            return


def set_debug(*args: Union[str, List[Union[str, int, EllipsisType]]], callback=None) -> None:
    """Print matrices whose name correspond to the pattern

    Examples:
        set_debug() will print nothing.
        set_debug(()) will print everthing possible.
        set_debug(1) will print everything happening the first layer.
        set_debug([1, 2, "head"]) will print the output of the second head of the first layer.
        set_debug([1, ..., "q"]) will print all queries of the first layer.
        set_debug([..., ..., "q"], [..., ..., "k"]) will print all queries and keys of all layers.

    Optionnally, a callback can be provided to perfom an action instead of printing.
    """

    args = [a if isinstance(a, tuple) else (a,) for a in args]
    DEBUG.clear()
    DEBUG.update(args)
    global DEBUG_CALLBACK
    DEBUG_CALLBACK = callback

@contextmanager
def temp_debug(*args: Union[str, List[Union[str, int, EllipsisType]]], callback) -> None:
    """Same as set_debug but only for the duration of the context."""
    old_debug = DEBUG.copy()
    old_callback = DEBUG_CALLBACK
    set_debug(*args, callback=callback)
    try:
        yield
    finally:
        set_debug(*old_debug, callback=old_callback)


def chrange(x, start_range, target_range, flip=False):
    normalised = (x - start_range[0]) / (start_range[1] - start_range[0])
    if flip:
        normalised = 1 - normalised
    return normalised * (target_range[1] - target_range[0]) + target_range[0]

def pprint_2d_tensor(t: TensorType):
    if t.dim() == 3:
        t = t.squeeze(0)

    if t.dim() != 2:
        return print(t)

    maxi = t.max().item()
    mini = t.min().item()

    def color(x):
        if x < 0:
            v = int(chrange(x, (mini, 0), (0, 180), flip=True))
            c = f'{v//2};{v//2};{v}'
        elif maxi == 0:
            c = '0;0;0'
        else:
            v = int(chrange(x, (0, maxi), (0, 180), flip=False))
            c = f'{v//2};{v};{v//2}'

        return f'\033[48;2;{c}m'

    width = len(f"{maxi:.1f}")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t[i, j].item()
            print(f"{color(v)}{v:{width}.1f}", end=' ')
        print("\033[0m")



def show_model(model: torch.nn.Module) -> None:
    params = list(model.named_parameters())
    param_count = len(params)

    width = math.ceil(math.sqrt(param_count))
    height = math.ceil(param_count / width)

    plt.figure(figsize=(width * 2, height * 2))
    for i, (name, param) in enumerate(params):
        plt.subplot(height, width, i + 1)
        if param.ndim == 1:
            plt.plot(param.detach().numpy())
        else:
            plt.imshow(param.detach().numpy())
            # Show scale
            plt.colorbar()
        plt.title(name)

    plt.tight_layout()

def show_transformer(model: Transformer) -> None:
    height = (model.heads + 1) * model.depth + 2

    def imshow(x, line, col, name: str):
        # plt.subplot(height, 3, line * 3 + col + 1)
        plt.subplot(3, height, col * height + line + 1)
        plt.imshow(x.detach().numpy())
        plt.colorbar()
        plt.title(name)


    imshow(model.embedding.weight, 0, 1, "Embedding")
    for l, layer in enumerate(model.blocks):
        head: AttentionHead
        for h, head in enumerate(layer.heads):
            imshow(head.queries, l * (model.heads + 1) + h + 1, 0, f"Layer {l} Head {h} Q")
            imshow(head.keys,    l * (model.heads + 1) + h + 1, 1, f"Layer {l} Head {h} K")
            imshow(head.values,  l * (model.heads + 1) + h + 1, 2, f"Layer {l} Head {h} V")
        imshow(layer.weight, l * (model.heads + 1) + model.heads + 1, 0, f"Layer {l} Weight")
    imshow(model.unembedding, height - 1, 1, "Unembedding")

    plt.tight_layout()


def show_activations(model: Transformer, input_: TensorType['token']) -> None:
    activations = []
    def store_activations(value: TensorType, *name: Union[str, int]):
        activations.append((' '.join(map(str, name)), value))

    with temp_debug((), callback=store_activations):
        model(input_)

    param_count = len(activations)
    width = math.ceil(math.sqrt(param_count))
    height = math.ceil(param_count / width)

    fig, axes = plt.subplots(height, width)

    for i, (name, param) in enumerate(activations):
        plt.subplot(height, width, i + 1)
        param.squeeze_(0)
        ax = axes[i // width, i % width]
        if param.ndim == 1:
            ax.stem(param.detach().numpy())
        else:
            im = ax.imshow(param.detach().numpy())
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    value = param[i, j].item()
                    if abs(value) > 0.01:
                        ax.text(j, i, f"{param[i, j]:.1f}", ha="center", va="center", color="w")
            # im.colorbar()  # Show scale
        plt.title(name)

    plt.tight_layout()



##########################
#       Other tools      #
##########################


def cat(*args: Union[torch.TensorType, list]):
    """Concatenate 2D tensors on both dimensions.

    Args are stacked vertically, and each arg is stacked horizontally."""
    parts = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            parts.append(arg)
            # print(arg.shape)
        else:
            # print(*[a.shape for a in arg])
            parts.append(torch.cat(arg, dim=1))


    return torch.cat(parts, dim=0)
