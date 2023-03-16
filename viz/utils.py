import pygame
from pygame import Rect, Vector2 as Vec2
import torch
from .engine import *

from hand import ResidualMLP, Transformer, AttentionHead, TransformerBlock, MultiAttentionHead
from viz.engine.gfx import GFX


def to_new_implem(model: Transformer) -> Transformer:
    emb_size = model.embedding.weight.shape[1]
    new = Transformer(model.voc_size, emb_size, model.depth, model.heads,
                      model.position_encoder, model.mlp_dims)
    new.embedding = model.embedding
    new.unembedding = model.unembedding
    new.blocks = torch.nn.Sequential(*[
        TransformerBlock(emb_size, model.heads, model.mlp_dims, i)
        for i in range(model.depth)
    ])

    for i, block in enumerate(model.blocks):
        if model.mlp_dims is not None:
            if i % 2 == 0:
                new.blocks[i // 2].attention = block
            else:
                new.blocks[i // 2].mlp = block
        else:
            new.blocks[i].attention = block

    return new


def transformer_outline(gfx: GFX, model: Transformer, pos=(50, 50)) -> Rect:
    pos = Vec2(pos)
    rect = Rect(pos, (0, 0))

    # Draw the embedding with its name above
    r = draw_matrix(gfx, model.embedding.weight,
        grad=model.embedding.weight.grad,
        topleft=pos, title="Embedding")
    rect.union_ip(r)
    # Draw the pos embedding on the righ
    r = draw_matrix(gfx, model.position_encoder,
        grad=model.position_encoder.grad,
        topleft=(r.right + 20, pos.y), title="Positional embedding")
    rect.union_ip(r)

    # Draw the blocks
    for i, block in enumerate(model.blocks):
        r = transformer_block_outline(gfx, block, pos=(pos.x, r.bottom + 40))
        gfx.text(f"Layer {i}", 15, "white", midbottom=r.midtop)
        rect.union_ip(r)

    # Draw the unembedding
    r = draw_matrix(gfx,
                    model.unembedding,
                    topleft=(pos.x, r.bottom + 20),
                    title="Unembedding")
    rect.union_ip(r)

    return rect


def transformer_block_outline(gfx: GFX, block: TransformerBlock,
                              pos=(50, 50)) -> Rect:
    # Draw each attention head
    r = Rect(pos, (0, 0))
    attention: AttentionHead
    for i, attention in enumerate(block.attention.heads):
        out_matrix = block.attention.weight[i *
                                            attention.values.shape[1]:(i + 1) *
                                            attention.values.shape[1]]
        r = attention_outline(gfx,
                              attention,
                              out_matrix=out_matrix,
                              pos=r.topright + Vec2(20 * (i > 0), 0),
                              name=i)
    r.union_ip(Rect(pos, (0, 0)))

    # Draw the residual MLP
    if block.mlp is not None:
        r2 = mlp_outline(gfx, block.mlp, pos=r.bottomleft)

    return r.union(r2)


def attention_outline(gfx: GFX,
                      attention: AttentionHead,
                      out_matrix=None,
                      pos=(50, 50),
                      name="") -> Rect:
    name = str(name)
    # Draw QK matrix
    r = draw_matrix(gfx,
                    attention.queries @ attention.keys.T,
                    topleft=pos,
                    title="QK " + name)
    # Draw OV matrix below
    ov = attention.values @ out_matrix if out_matrix is not None else attention.values
    r2 = draw_matrix(gfx, ov, topleft=r.bottomleft, title="OV " + name)

    return r.union(r2)

def mlp_outline(gfx: GFX, mlp: ResidualMLP, pos) -> pygame.Rect:
    r = Rect(pos, (0, 0))
    for i, layer in enumerate(mlp.layers):
        # Draw the weight matrix
        r = draw_matrix(gfx,
                        layer.weight,
                        grad=layer.weight.grad,
                        topleft=r.topright + Vec2(20 * (i > 0), 0),
                        title=f"Weight {i}")
        # Draw the bias vector on the right
        r = draw_matrix(gfx,
                        layer.bias,
                        grad=layer.bias.grad,
                        topleft=r.topright + Vec2(20, 0),
                        title=f"Bias {i}")

    return r.union(Rect(pos, (0, 0)))


def draw_matrix(gfx: GFX,
                matrix: torch.Tensor,
                size=20,
                title="",
                legend=False,
                only_return_rect=False,
                grad: torch.Tensor=None,
                **anchor) -> Rect:
    """Draw a matrix or vector as a heatmap"""
    m = matrix.detach()
    if m.dim() == 1:
        m = m.unsqueeze(1)

    # Layout
    heatmap_rect = Rect(0, 0, m.shape[1] * size, m.shape[0] * size)
    full_rect = heatmap_rect.copy()
    if title:
        title_height = font().size(title)[1]
        full_rect.height += title_height
        heatmap_rect.y += title_height
    if legend:
        full_rect.width += size
        heatmap_rect.x += size
    anchor, anchor_pos = anchor.popitem()
    setattr(full_rect, anchor, anchor_pos)
    heatmap_rect.topleft += Vec2(full_rect.topleft)

    # End early if only the rect is needed
    if only_return_rect:
        return full_rect

    if title:
        gfx.text(title, midtop=full_rect.midtop)

    screen_size = size * gfx.screen_scale

    def fmt(v):
        if screen_size < 15:
            return ""
        f = f"{v:.{int(screen_size // 30)}f}"
        # If only zeros, don't show them
        if all(c == "0" for c in f if c.isdigit()):
            return ""
        return f

    # Use a symmetric range for the colors
    value_range = max(abs(m.min()), abs(m.max())).item()
    if value_range == 0:
        value_range = 1

    if grad is not None:
        grad_range = max(abs(grad.min()), abs(grad.max())).item()
        if grad_range == 0:
            grad_range = 1
        if grad.ndim == 1:
            grad = grad.unsqueeze(1)


    def color(v, range_):
        zero_color = (20, 30, 40)
        if v < 0:
            return mix(RED, zero_color, chrange(v, (-range_, 0), (0, 1)))
        else:
            return mix((zero_color), GREEN, chrange(v, (0, range_), (0, 1)))

    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            v = m[r, c].item()
            rect = Rect(heatmap_rect.x + c * size, heatmap_rect.y + r * size,
                        20, 20)
            gfx.box(rect, color(v, value_range))
            if grad is not None:
                g = -grad[r, c].item()
                gfx.rect(*rect, color(g, grad_range), 3)
            gfx.text(
                fmt(v),
                15 / gfx.screen_scale,
                "white",
                center=rect.center,
            )

    gfx.rect(*heatmap_rect, "pink", 1)
    return full_rect