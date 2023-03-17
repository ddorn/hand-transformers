from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
import math
import random
from textwrap import dedent
from typing import Callable, List, Optional, Tuple, Type, Union, Set, Dict

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm
from typeguard import typechecked

TT = TensorType

patch_typeguard()

# ------------------------- #
# Reimplementation of GPT #
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


class AttentionHead(torch.nn.Module):

    def __init__(self,
                 embedding_size: int,
                 out_size: int,
                 layer: int = None,
                 n: int = None) -> None:
        """A single attention head.

        Args:
            embedding_size (int): The size of the input embedding.
            out_size (int): The size of the output embedding.
            layer (int): The layer number, for debugging purposes.
            n (int): The head number, for debugging purposes.
        """
        super().__init__()

        self.layer = layer
        self.n = n

        self.queries = torch.nn.Parameter(torch.randn(embedding_size,
                                                      out_size))
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

    def set(self, keys, queries, values):
        """Helper function to set the three matrices. Checks that the sizes are correct."""
        assert keys.shape == self.keys.shape
        assert queries.shape == self.queries.shape
        assert values.shape == self.values.shape
        self.keys = torch.nn.Parameter(keys)
        self.queries = torch.nn.Parameter(queries)
        self.values = torch.nn.Parameter(values)

    def qk(self):
        """Returns the qk matrix."""
        return self.queries @ self.keys.T


class MultiAttentionHead(torch.nn.Module):

    def __init__(self,
                 embedding_size: int,
                 heads: int,
                 layer: int = None) -> None:
        assert embedding_size % heads == 0, "embedding size must be divisible by heads."
        super().__init__()
        self.layer = layer

        out_size = embedding_size // heads

        self.heads = torch.nn.ModuleList([
            AttentionHead(embedding_size, out_size, layer=layer, n=n)
            for n in range(heads)
        ])
        self.weight = torch.nn.Parameter(
            torch.randn(embedding_size, embedding_size))

    def forward(self, x: TensorType['b', 't',
                                    'emb']) -> TensorType['b', 't', 'out']:
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        debug(combined, self.layer, "heads-combined")
        multihead = combined @ self.weight
        debug(multihead, self.layer, "multihead")
        output = multihead + x
        debug(output, self.layer, "layer")
        return output


class ResidualMLP(torch.nn.Module):

    def __init__(self,
                 embeding_size: int,
                 *layers_dims: int,
                 layer: int = None) -> None:
        """An MLP with residual a connection.

        Args:
            embedding_size (int): The size of the input and output embedding.
            *layers_dims (int): The dimensions of the hidden layers.
            layer (int): The layer number, for debugging purposes.
        """

        super().__init__()
        self.layer = layer
        dims = embeding_size, *layers_dims, embeding_size
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])

    def forward(
        self, x: TensorType['batch', 'tokens', 'embedding_size']
    ) -> TensorType['batch', 'tokens', 'embedding_size']:
        initial = x
        for l, layer in enumerate(self.layers):
            x = layer(x)
            debug(x, self.layer, "mlp", l)
            x = torch.relu(x)
            debug(x, self.layer, "mlp", l, "relu")

        x = x + initial
        debug(x, self.layer, "mlp", "residual")
        return x


class TransformerBlock(torch.nn.Module):

    def __init__(self,
                 embedding_size: int,
                 heads: int,
                 mlp_dims: Optional[Tuple[int, ...]] = None,
                 layer: int = None) -> None:
        """A single transformer block.

        Args:
            embedding_size (int): The size of the input and output embedding.
            heads (int): The number of attention heads.
            mlp_dims (Optional[Tuple[int, ...]]): The dimensions of the hidden layers in the MLP.
            layer (int): The layer number, for debugging purposes.
        """
        super().__init__()
        self.layer = layer

        self.attention = MultiAttentionHead(embedding_size, heads, layer=layer)
        if mlp_dims is not None:
            self.mlp = ResidualMLP(embedding_size, *mlp_dims, layer=layer)
        else:
            self.mlp = None

    def forward(
        self, x: TensorType['batch', 'tokens', 'embedding_size']
    ) -> TensorType['batch', 'tokens', 'embedding_size']:
        x = self.attention(x)
        if self.mlp is not None:
            x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):

    def __init__(self,
                 voc_size: int,
                 embedding_size: int,
                 depth: int,
                 heads: int,
                 max_prompt_len: int,
                 mlp_dims: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()

        self.depth = depth
        self.heads = heads
        self.mlp_dims = mlp_dims
        self.voc_size = voc_size
        self.embedding = torch.nn.Embedding(voc_size, embedding_size)
        self.position_encoder = torch.nn.Parameter(
            torch.randn(max_prompt_len, embedding_size))
        self.blocks = torch.nn.Sequential(*[
            TransformerBlock(embedding_size, heads, mlp_dims, layer=i)
            for i in range(depth)
        ])
        self.unembedding = torch.nn.Parameter(
            torch.rand(embedding_size, voc_size))

    def forward(self, x: TensorType['batch', 'token']) -> List[str]:
        debug(x, "input")
        embeded = self.embedding(x)
        debug(embeded, "embeded")
        # with_pos = self.position_encoder(embeded)
        with_pos = embeded + self.position_encoder
        debug(with_pos, "embed+pos")
        x = self.blocks(with_pos)
        out = x[:, -1, :].squeeze(1)  # only the last token is the prediction
        unembeded = out @ self.unembedding
        debug(unembeded, "unembeded")
        return unembeded
        # probas = torch.softmax(unembeded, dim=-1)
        # debug(probas, "probas")
        # return probas

    def loss(self, x: TensorType['batch', 'token'],
             y: TensorType['batch']) -> TensorType['batch']:
        logits = self.forward(x)
        return torch.nn.functional.cross_entropy(logits, y)

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                # not zero
                torch.nn.init.normal_(p, std=0.01)


# ----------------------- #
# Framework for exercises #
# ----------------------- #


class Task:
    char_to_token: Dict[str, int]
    tokens: List[str]
    prompt_len: int

    def __init__(self,
                 tokens: list[str],
                 prompt_len: int,
                 pad_token=None) -> None:
        self.tokens = tokens
        self.prompt_len = prompt_len
        self.pad_token = pad_token
        if pad_token is not None:
            self.tokens.append(pad_token)
        self.char_to_token = {char: i for i, char in enumerate(tokens)}

    def generate(self, n: int) -> tuple[list[str], list[str]]:
        raise NotImplementedError

    def generate_batch(
            self, batch_size: int
    ) -> tuple[TT['batch', 'token', int], TT['batch', int]]:
        prompts, answers = self.generate(batch_size, False)

        if self.pad_token is None and not all(
                len(prompt) == self.prompt_len for prompt in prompts):
            raise ValueError(
                "Some prompts are not of the same length, and no padding token is defined."
            )

        # encode prompts and answers
        return self.encode(prompts, self.prompt_len), self.encode(answers).squeeze(1)

    def encode(self, prompts: List[str], pad_to: int = None) -> TT['prompt', 'token', int]:
        if pad_to is not None and not all(len(prompt) == pad_to for prompt in prompts):
            assert self.pad_token is not None, "No padding token defined but prompts are not of the same length."
            prompts = [
                self.pad_token * (pad_to - len(prompt)) + prompt
                for prompt in prompts
            ]

        return torch.tensor([[self.char_to_token[char] for char in prompt]
                            for prompt in prompts])


    def decode(self, tokens: TT['prompt', 'token', int]) -> List[str]:
        if tokens.ndim == 1:
            return [self.tokens[token] for token in tokens]
        return [
            ''.join(self.tokens[token] for token in prompt)
            for prompt in tokens
        ]

    def test(self, model: Transformer, verbose: int = 0, nb_tests: tinn=100) -> float:
        """Test the model on this task.

        Args:
            model (Transformer): The model to test.
            verbose (int):
                0: No output.
                1: Print the prompts and answers that are wrong.
                2: Print all prompts and answers.
                3: Print the prompts and answers that are wrong, and the probas.

        Returns:
            float: The accuracy of the model on this task.
        """
        with torch.no_grad():
            prompts, answers = self.generate(nb_tests)
            xs = self.encode(prompts, self.prompt_len)
            probas = model(xs)
            predictions = torch.argmax(probas, dim=-1)
            predictions = self.decode(predictions)

            for prompt, answer, prediction in zip(prompts, answers, predictions):
                if verbose == 2 or (verbose == 1 and answer != prediction):
                    print(f"{prompt} → {answer} ({prediction})")
                elif verbose == 3:
                    pass
                elif verbose > 3:
                    raise ValueError("verbose should be in [0, 3].")

            return sum(1
                       for answer, prediction in zip(answers, predictions)
                       if answer == prediction) / len(answers)

    def loss(self, model: Transformer) -> float:
        """Compute the loss of the model on this task.

        Args:
            model (Transformer): The model to test.

        Returns:
            TensorType['batch']: The loss of the model on this task.
        """
        probas = model(self.tokenized_prompts)
        return torch.nn.functional.cross_entropy(
            probas, self.tokenized_answers.squeeze(1)).item()

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}"
        prompts, answers = self.generate(10)
        if len(prompts) <= 10:
            examples = range(len(prompts))
        else:
            examples = range(10)

        for i in examples:
            s += f"\n  {prompts[i]} → {answers[i]}"

        # Add tokens with their IDs
        s += f"\nTokens: {self.tokens}"
        return s


class StaticTask(Task):
    """A task is a finite set of prompts and 1-char answers."""
    prompts: List[str]
    answers: List[str]

    def __init__(self, texts: list[str]) -> None:
        """Create a task from pre-defined set of prompts where only the last char is to guess.

        Args:
            texts (List[str]): A list of texts, each text is a concatenation of
                a prompt and a 1-char answer. Each prompt must have the same
                length.
        """
        assert len(texts) > 0
        assert all(len(t) == len(texts[0]) for t in texts)

        self.prompts = [text[:-1] for text in texts]
        self.answers = [text[-1] for text in texts]
        tokens = list(sorted(set(''.join(texts))))
        super().__init__(tokens)

    def generate(self, n: int) -> tuple[list[str], list[str]]:
        if n >= len(self.prompts):
            return self.prompts, self.answers
        else:
            indices = random.sample(range(len(self.prompts)), n)
            return [self.prompts[i]
                    for i in indices], [self.answers[i] for i in indices]

    @cached_property
    def max_prompt_len(self) -> int:
        return max(len(prompt) for prompt in self.prompts)

    @cached_property
    def tokenized_prompts(self) -> TT['prompt', 'token', int]:
        return self.encode(self.prompts)

    @cached_property
    def tokenized_answers(self) -> TT['prompt', 'token', int]:
        return self.encode(self.answers)


# ------------------------------ #
#       Training functions       #
# ------------------------------ #


@dataclass
class Perfs:
    """A class to store the performance of a model during training."""
    loss: List[float]
    accuracy: Optional[List[float]] = None
    models: Optional[List[Transformer]] = None

    def update(self, lost, model: Transformer, task: Task):
        self.loss.append(lost.item())
        if self.accuracy is not None:
            accuracy = task.test(model)
            self.accuracy.append(accuracy)
        if self.models is not None:
            self.models.append(deepcopy(model))

    def plot_loss(self):
        plt.plot(self.loss[1:])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    def plot_accuracy(self):
        plt.plot(self.accuracy)


def train(
    model: Transformer,
    task: Task,
    epochs=1,
    optimizer=None,
    perfs: Optional[Perfs] = None,
):

    loss = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.01,
                                    weight_decay=0.001)

    if perfs is None:
        perfs = Perfs([])

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(task.tokenized_prompts)
        lost = loss(preds, task.tokenized_answers.squeeze())
        lost.backward()
        optimizer.step()

        # Update metrics
        perfs.update(lost, model, task)

        # if epoch % (epochs // 10) == (epochs // 10) - 1 or epoch == 0:
        #     print(f"Epoch {epoch+1} loss: {lost}")

    return perfs


# ------------------------------ #
# Help for visualizing the model #
# ------------------------------ #

DEBUG = set()  # debug nothing
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


def set_debug(*args: Union[str, List[Union[str, int, EllipsisType]]],
              callback=None) -> None:
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

    args = [a if isinstance(a, tuple) else (a, ) for a in args]
    DEBUG.clear()
    DEBUG.update(args)
    global DEBUG_CALLBACK
    DEBUG_CALLBACK = callback


@contextmanager
def temp_debug(*args: Union[str, List[Union[str, int, EllipsisType]]],
               callback) -> None:
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


def model_diff(model1: Transformer, model2: Transformer) -> None:
    """Return a model whose parameters are the difference between the two models."""

    model = deepcopy(model1)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        p1.data = p2.data - p1.data
    return model


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


def show_transformer(model: Transformer, scale=4) -> None:
    heads = (model.heads + 1) * model.depth
    mlps_size = model.depth * 2 if len(model.blocks) != model.depth else 0
    height = heads + mlps_size + 2

    # plt.figure(figsize=(height * 2, 3 * 2))
    fig, axes = plt.subplots(3, height, figsize=(height * scale, 3 * scale))
    axes = axes.T
    show_matrix(model.embedding.weight, axes[0, 1], "Embedding")
    show_matrix(model.position_encoder, axes[0, 2], "Position encoder")
    line = 1
    for b, block in enumerate(model.blocks):
        if isinstance(block, MultiAttentionHead):
            for h, head in enumerate(block.heads):
                show_matrix(head.queries, axes[line, 0],
                            f"Layer {head.layer} Head {h} Q")
                show_matrix(head.keys, axes[line, 1],
                            f"Layer {head.layer} Head {h} K")
                show_matrix(head.values, axes[line, 2],
                            f"Layer {head.layer} Head {h} V")
                line += 1
            show_matrix(block.weight, axes[line, 1], f"Layer {b} Weight")
            line += 1
        elif isinstance(block, ResidualMLP):
            for i, layer in enumerate(block.layers):
                show_matrix(layer.weight, axes[line, i],
                            f"Layer {b} MLP layer {i}")
                axes[line + 1, i].stem(layer.bias.detach().numpy())
                axes[line + 1, i].set_title(f"Layer {b} MLP bias {i}")
            line += 2
    show_matrix(model.unembedding, axes[line, 1], "Unembedding")


def get_activations(
        model: Transformer,
        prompt: TensorType["batch", "token"]) -> Dict[str, TensorType]:
    """Return a dictionary of all activations of the model."""

    activations = {}

    def store_activations(value: TensorType, *name: Union[str, int]):
        activations[' '.join(map(str, name))] = value

    with torch.no_grad():
        with temp_debug((), callback=store_activations):
            model(prompt)

    return activations


def show_activations(model: Transformer, input_: TensorType['token']) -> None:
    assert len(input_.shape) == 2 and input_.shape[
        0] == 1, "show_activations only works with a single input"

    activations = get_activations(model, input_)

    with torch.no_grad():

        param_count = len(activations)
        width = math.ceil(math.sqrt(param_count))
        height = math.ceil(param_count / width)

        fig, axes = plt.subplots(height,
                                 width,
                                 figsize=(width * 4, height * 4))

        for i, (name, param) in enumerate(activations.items()):
            # plt.subplot(height, width, i + 1)
            param.squeeze_(0)
            ax = axes[i // width, i % width]
            if param.ndim == 1:
                ax.stem(param.detach().numpy())
                ax.set_title(name)
            else:
                show_matrix(param, ax, name)


def show_batch(tensor: TensorType["batch", "token", "dim"],
               prompts: TensorType["batch", "token", "position"],
               width: int = None,
               **kwargs) -> None:

    if width is None:
        width = math.ceil(math.sqrt(tensor.shape[0]))

    height = math.ceil(tensor.shape[0] / width)

    fig, axs = plt.subplots(
        height,
        width,
        squeeze=False,
        figsize=(width * tensor.shape[2] / 2, height * tensor.shape[1] / 2),
        sharex=True,
        sharey=True,
    )

    for i, prompt in enumerate(prompts):
        show_matrix(tensor[i, :, :], axs[i // width, i % width],
                    " ".join(map(str, prompt.tolist())), **kwargs)


def show_matrix(x, axis=None, title: str = "", **kwargs):
    if axis is None:
        axis = plt.gca()

    x = x.detach().numpy()
    im = axis.imshow(x, **kwargs)
    colors = im.cmap(im.norm(x))
    max_shown = np.abs(x).max() * 0.01

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            value = x[i, j].item()
            if abs(value) < max_shown:
                continue
            elif abs(value - int(value)) < 0.05:
                text = str(int(value))
            else:
                text = f"{value:.2f}"

            if text != "0":
                color = 'white' if colors[i, j, :4].mean() < 0.5 else 'black'
                axis.text(j, i, text, ha="center", va="center", color=color)
    # plt.colorbar()
    if title:
        axis.set_title(title)


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


def pad(x, width=None, height=None, left=0, top=0):
    """Add zeros arounds a 2D tensor so that its size is [bottom, right]
    and its top-left corner is at [top, left]."""
    if isinstance(x, list):
        x = torch.tensor(x)
    if width is None:
        width = x.shape[1]
    if height is None:
        height = x.shape[0]

    x = cat(torch.zeros(top, width), [
        torch.zeros(x.shape[0], left), x,
        torch.zeros(x.shape[0], width - left - x.shape[1])
    ], torch.zeros(height - top - x.shape[0], width))
    return x


def copy(in_dims: List[int], out_dims: List[int], shape: Tuple[int, int]):
    """
    Create a matrix that moves the values of the input dims to the output dims
    when applied on the right of a vector (input * Matric => permuted).
    """

    assert len(in_dims) == len(out_dims)

    matrix = torch.zeros(*shape)
    for in_dim, out_dim in zip(in_dims, out_dims):
        matrix[in_dim, out_dim] = 1
    return matrix


def select(dims: List[int], value: List[int], shape=Tuple[int, int], scale=10):
    """
    Create a keys and a queries matrix that selects vectors with higher dot
    product with the value vector.
    This assumes that that the average values of every token in the [dims]
    is higher than zero. If not, the scale parameter needs to be negative.
    """
    assert len(dims) == len(value)

    keys = torch.zeros(*shape)
    keys[dims, 0] = torch.tensor(value, dtype=keys.dtype)

    queries = torch.zeros(*shape)
    queries[dims, 0] = scale

    return keys, queries


def attend_abs_position(dims: List[int],
                        permutation: List[int],
                        scale=10,
                        shape=Tuple[int, int]):
    """
    Create a keys and queries matrix that selects token at the position in the permutation.
    """
    assert len(dims) == len(permutation)
    keys = torch.zeros(*shape)
    keys[dims, list(range(len(dims)))] = scale
    queries = torch.zeros(*shape)
    queries[dims, permutation] = 1
    return keys, queries


##########################
#    Modifying models    #
##########################


def perturb(model, noise_strengh: float, prop: bool = True):
    """Modify the weight of a model in place by adding a random noise.

    Args:
        model: The model to modify.
        noise_strengh: The standard deviation of the noise.
        prop: If True, the noise is proportional to the weight.
    """
    with torch.no_grad():
        for p in model.parameters():
            perturbation = torch.randn_like(p)
            if prop:
                p += noise_strengh * perturbation * p.abs()
            else:
                p += noise_strengh * perturbation
    return model
