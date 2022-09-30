import random
from typing import Dict, List
from torch import Tensor
from hand import Exercise, mkexo, set_debug

__all__ = ['EXOS', 'set_debug']

# Solved. Trivial.
@mkexo(name="LastChar", voc="abc", input_len=3)
def last_char():
    """Complete the text by repeating the last character."""

    size = random.randrange(1, 4)
    s = "".join(random.choices("abc", k=size))
    return s, s[-1]


# Solved. Simple.
@mkexo(name="CycleTwo", voc="abc", input_len=3)
def cycle_two():
    """Complete the text by repeating the second-last character."""

    size = random.randrange(2, 4)
    s = "".join(random.choices("abc", k=size))
    return s, s[-2]


# Solved. Somewhat simple.
@mkexo(name="FirstChar", voc="abc", input_len=5)
def first_char():
    """Complete the text by repeating the first character.
    Note: the first character is not always at the same position,
    since inputs have variable length."""

    size = random.randrange(2, 5)  # We kindly ensure that there is always one pad token
    s = "".join(random.choices("abc", k=size))
    return s, s[0]


# Doable.  abc|cb
@mkexo(name="Reverse", voc="abc|", input_len=8)
def reverse():
    """Complete the text by reversing the input after the bar "|"."""

    size = random.randrange(1, 5)
    s = "".join(random.choices("abc", k=size))
    s = s + "|" + s[::-1]
    cut = random.randrange(s.index("|") + 1, len(s))
    return s[:cut], s[cut]


# Easy but not very interesting. Can be nice to get started.
@mkexo(name="Difference", voc="01", input_len=2)
def difference():
    """Complete by 0 if the two digits are different and by 1 if they are the same."""

    a, b = random.choices("01", k=2)
    return a + b, "01"[a == b]


@mkexo(name="AllTheSame", voc="012", input_len=3)
def all_the_same():
    """Complete by 1 if all the digits are the same and by 0 otherwise."""

    a, b, c = random.choices("012", k=3)
    return a + b + c, "01"[a == b == c]


@mkexo(name="KinderAdder", voc="01234", input_len=2)
def kinder_adder():
    """Complete by the sum of the two digits.
    Note: no input will use digits 3 and 4."""

    a, b = random.choices("012", k=2)
    return a + b, str(int(a) + int(b))


# Probaby not the most interesting
@mkexo(name="LengthParity", voc="0", input_len=8)
def length_parity():
    """Complete by 0 if the input length is even and by the empty token otherwise."""

    size = random.randrange(1, 8)
    return "0" * size, "0" * (size % 2)


@mkexo(name="Min", voc="0123", input_len=4)
def minimum():
    """Complete by the minimum of the four digits."""

    a, b, c, d = s = random.choices("0123", k=4)
    return ''.join(s), min(a, b, c, d)


# Essential. Basis for many other...
@mkexo(name="Copy", voc="01|", input_len=7)
def copy():
    """Copy the input after the bar "|"."""

    size = random.randrange(1, 4)
    s = "".join(random.choices("01", k=size))
    s = s + "|" + s
    cut = random.randrange(s.index("|") + 1, len(s))
    return s[:cut], s[cut]

# Medium. Builds on FirstChar.
@mkexo(name="Induction", voc="ABCDE", input_len=8)
def induction():
    """Complete with the token that followed the last time the last token was pressent.
    For instance if the input ends in A, complete with the token after the last A."""

    size = random.randrange(3, 8)
    s = ''.join(random.choices("ABCDE", k=size - 1))
    s += random.choice(s)
    return s, s[s[:-1].rindex(s[-1]) + 1]


@mkexo(name="BinaryAdd", voc="01+=", input_len=12)
def binary_add():
    """Complete with the sum of the two binary numbers."""

    size = 3
    a = random.randrange(0, 2 ** size)
    b = random.randrange(0, 2 ** size)
    s = f"{a:b}+{b:b}={a + b:b}"

    cut = random.randrange(s.index("=") + 1, len(s))
    return s[:cut], s[cut]


EXOS: Dict[str, Exercise] = {
    e.name: e
    for e in globals().values()
    if isinstance(e, Exercise)
}

if __name__ == '__main__':
    import sys
    print(sys.argv, len(sys.argv))

    for name, exo in EXOS.items():
        print(f'\033[34m{name}\033[0m', end=': ')
        print(exo.description.splitlines()[0])
