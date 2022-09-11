import random
from typing import List
from torch import Tensor
from hand import Exercise, mkexo

__all__ = [
    'exo0', 'exo1', 'exo2', 'exo3', 'exo4', 'exo5', 'exo6', 'exo7', 'exo8',
    'exo9', 'exo10', 'EXOS'
]

@mkexo(name="00-LastChar", voc="abc", input_len=3)
def exo0():
    """Complete the text by repeating the last character."""

    size = random.randrange(1, 4)
    s = "".join(random.choices("abc", k=size))
    return s, s[-1]


@mkexo(name="01-CycleTwo", voc="abc", input_len=3)
def exo1():
    """Complete the text by repeating the second-last character."""

    size = random.randrange(2, 4)
    s = "".join(random.choices("abc", k=size))
    return s, s[-2]


@mkexo(name="02-FirstChar", voc="abc", input_len=3)
def exo2():
    """Complete the text by repeating the first character.
    Note: the first character is not always at the same position,
    since inputs have variable length."""

    size = random.randrange(1, 4)
    s = "".join(random.choices("abc", k=size))
    return s, s[0]


@mkexo(name="03-Reverse", voc="abc|", input_len=5)
def exo3():
    """Complete the text by reversing the input after the bar "|"."""

    size = random.randrange(1, 4)
    s = "".join(random.choices("abc", k=size))
    s = s + "|" + s[::-1]
    cut = random.randrange(s.index("|") + 1, len(s))
    return s[:cut], s[cut]


@mkexo(name="04-Difference", voc="01", input_len=2)
def exo4():
    """Complete by 0 if the two digits are different and by 1 if they are the same."""

    a, b = random.choices("01", k=2)
    return a + b, "01"[a == b]


@mkexo(name="05-AllTheSame", voc="012", input_len=3)
def exo5():
    """Complete by 1 if all the digits are the same and by 0 otherwise."""

    a, b, c = random.choices("012", k=3)
    return a + b + c, "01"[a == b == c]


@mkexo(name="06-KinderAdder", voc="01234", input_len=2)
def exo6():
    """Complete by the sum of the two digits.
    Note: no input will use digits 3 and 4."""

    a, b = random.choices("012", k=2)
    return a + b, str(int(a) + int(b))


@mkexo(name="07-LengthParity", voc="0", input_len=8)
def exo7():
    """Complete by 0 if the input length is even and by the empty token otherwise."""

    size = random.randrange(1, 8)
    return "0" * size, "0" * (size % 2)


@mkexo(name="08-Min", voc="0123", input_len=4)
def exo8():
    """Complete by the minimum of the four digits."""

    a, b, c, d = s = random.choices("0123", k=4)
    return ''.join(s), min(a, b, c, d)


@mkexo(name="09-ARecall", voc="A01", input_len=6)
def exo9():
    """Complete with the token following the last A."""

    size = random.randrange(2, 6)
    pat = "A" + random.choice("01")
    pred = random.choices("A01", k=random.randint(0, size - 2))
    after = random.choices("01", k=size - 2 - len(pred))
    s = ''.join(pred) + pat + ''.join(after)
    return s, pat[1]


@mkexo(name="10-TODO", voc="1234567890+=", input_len=6)
def exo10():
    """TODO"""


EXOS: List[Exercise] = [
    e for e in globals().values()
    if isinstance(e, Exercise)
]


def sol0():
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
    exo0.test_model(0, 1, 2, embedding, unembedding, pos_encoder, layers, 100)

def sol1():
    embedding = Tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    unembedding = Tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]])
    pos_encoder = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    layer_0_weight = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    layer_0_head_0_q = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_0_k = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_0_v = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_0 = [layer_0_head_0_q, layer_0_head_0_k, layer_0_head_0_v]

    layer_0_head_1_q = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_1_k = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_1_v = Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    layer_0_head_1 = [layer_0_head_1_q, layer_0_head_1_k, layer_0_head_1_v]

    layer_0_heads = [layer_0_head_0, layer_0_head_1]
    layer_0 = [layer_0_heads, layer_0_weight]

    layers = [layer_0]


if __name__ == '__main__':
    print(EXOS[1].print_template(1, 2, 3))
