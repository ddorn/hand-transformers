import random
from typing import Dict, List
from torch import Tensor
from hand import set_debug

__all__ = ['EXOS', 'set_debug']

def mkexo(*args, **kwargs):
    return None

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


@mkexo(name="XOR", voc="01", input_len=2)
def difference():
    """Complete by 1 if the two digits are different and by 0 if they are the same."""

    a, b = random.choices("01", k=2)
    return a + b, "01"[a != b]

@mkexo(name="IfThenElse", voc="01", input_len=3)
def if_then_else():
    """Input: XYZ. Complete by Y if X == 1 otherwise by Z."""
    s = "".join(random.choices("01", k=3))
    return s, s[1 + (s[0] == '0')]


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


# Doable. One new idea.
@mkexo(name="Sort", voc="|012345", input_len=12)
def sort():
    """Complete by the sorted version of the input after the bar."""

    size = random.randrange(2, 7)
    s = ''.join(random.choices("012345", k=size))
    complete = s + '|' + ''.join(sorted(s))
    cut = random.randrange(len(s) + 1, len(complete))
    return complete[:cut], complete[cut]


def fixed_addition(base: int, length: int):
    def to_base(x, n):
        return ((x == 0) and "0") or (to_base(x // n, n).lstrip("0") + "0123456789abcdefghijklmnopqrstuvwxyz"[x % n])

    a = random.randrange(0, base ** length)
    b = random.randrange(0, base ** length)
    s = to_base(a, base) + to_base(b, base)
    if length == 1:
        return s, to_base(a + b, base)[-1]
    else:
        d = random.randrange(0, length)
        return s + to_base(d, base), to_base(a + b, base).ljust(length+1, "0")[d+1]


@mkexo(name="AddFixedBase2Mod4", voc="01", input_len=5)
def add_1():
    """Input is of the form "d a₁ a₂ b₁ b₂. Complete by the d-th digit of a+b."""

    return fixed_addition(2, 2)

    size = 2
    a = random.randrange(0, 2 ** size)
    b = random.randrange(0, 2 ** size)
    d = random.randrange(0, 2)
    s = f"{d}{a:b}{b:b}"
    answer = f"{(a + b) % 4:b}"[d]

    return s, answer


@mkexo(name="BinaryAdd", voc="01+=", input_len=12)
def binary_add():
    """Complete with the sum of the two binary numbers."""

    size = 3
    a = random.randrange(0, 2 ** size)
    b = random.randrange(0, 2 ** size)
    s = f"{a:b}+{b:b}={a + b:b}"

    cut = random.randrange(s.index("=") + 1, len(s))
    return s[:cut], s[cut]

EXOS = {}
# EXOS: Dict[str, Exercise] = {
#     e.name: e
#     for e in globals().values()
#     if isinstance(e, Exercise)
# }

if __name__ == '__main__':
    import sys
    print(sys.argv, len(sys.argv))

    if len(sys.argv) >= 2:
        for name in sys.argv[1:]:
            print(EXOS[name])
    else:
        for name, exo in EXOS.items():
            print(f'\033[34m{name}\033[0m', end=': ')
            print(exo.description.splitlines()[0])
