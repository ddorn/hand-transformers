import random
from textwrap import dedent
from typing import Callable, List, Tuple

import einops
import matplotlib.pyplot as plt
import torch
from torchtyping import TensorType, patch_typeguard
import tqdm
from typeguard import typechecked

patch_typeguard()


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


class PositionEncoder(torch.nn.Module):
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

    def __init__(self, embedding_size: int, out_size: int) -> None:
        super().__init__()

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

        qk = torch.einsum("bti,bTi->btT", q, k)
        s = torch.softmax(qk / q.shape[-1]**0.5, dim=-1)
        out = torch.einsum("btT,bTi->bti", s, v)

        return out


class MultiAttentionHead(torch.nn.Module):

    def __init__(self, embedding_size: int, heads: int) -> None:
        assert embedding_size % heads == 0, "embedding size must be divisible by heads."
        super().__init__()

        out_size = embedding_size // heads

        self.heads = torch.nn.ModuleList(
            [AttentionHead(embedding_size, out_size) for _ in range(heads)])
        self.weight = torch.nn.Linear(embedding_size, embedding_size, bias=False)

    def forward(self, x: TensorType['b', 't',
                                    'emb']) -> TensorType['b', 't', 'out']:
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return x + self.weight(combined)


class AttentionOnlyTransformer(torch.nn.Module):

    def __init__(self, voc_size: int, embedding_size: int, depth: int,
                 heads: int, pos_encoder: TensorType['max_prompt_len', 'embedding_size']) -> None:
        super().__init__()

        self.depth = depth
        self.heads = heads
        self.voc_size = voc_size
        self.embedding = torch.nn.Embedding(voc_size, embedding_size)
        # self.position_encoder = PositionEncoder()
        self.position_encoder = pos_encoder
        self.blocks = torch.nn.Sequential(
            *[MultiAttentionHead(embedding_size, heads) for _ in range(depth)])
        self.unembedding = torch.nn.Parameter(torch.rand(embedding_size, voc_size))

    def forward(self, x: TensorType['batch', 'token']) -> List[str]:
        embeded = self.embedding(x)
        # with_pos = self.position_encoder(embeded)
        with_pos = embeded + self.position_encoder
        x = self.blocks(with_pos)
        out = x[:, -1, :].squeeze(1)  # only the last token is the prediction
        unembeded = out @ self.unembedding
        probas = torch.softmax(unembeded, dim=-1)
        return probas


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
    ) -> AttentionOnlyTransformer:

        # Shorthand
        p = lambda x: torch.nn.Parameter(x, requires_grad=False)

        model = AttentionOnlyTransformer(self.tokenizer.vocab_size,
                                         inner_size * heads, depth, heads, position_encoder)
        model.embedding.weight = p(embedding)
        model.unembedding = p(unembedding)

        for i, (heads, weight) in enumerate(layers):
            for j, (q, k, v) in enumerate(heads):
                model.blocks[i].heads[j].queries = p(q)
                model.blocks[i].heads[j].keys = p(k)
                model.blocks[i].heads[j].values = p(v)
            model.blocks[i].weight.weight = p(weight)

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

class SaveOutputs:
    """Hook to save outputs that removes itself after one use."""
    def __init__(self, name=""):
        self.name = name
        self.outputs = []

    def hook(self, module, inputs, outputs):
        print(self.name, module, "ouptuts:")
        print(outputs)
        self.outputs.append(outputs)

    def add_to(self, module: torch.nn.Module):

        def _hook(module, inputs, outputs):
            try:
                self.hook(module, inputs, outputs)
            finally:
                handle.remove()

        handle = module.register_forward_hook(_hook)

    def add_to_all(self, modules: list, type=None):
        for module in modules:
            if not type or isinstance(module, type):
                self.add_to(module)

        return self

def mkexo(name: str, voc: str, input_len: int) -> Exercise:

    def decorator(f: Callable[[], Tuple[str, str]]) -> Exercise:
        return Exercise(name, CharTokenizer(voc, input_len), f)

    return decorator