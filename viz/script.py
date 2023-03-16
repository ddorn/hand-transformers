import torch
from hand import Task, Transformer, TransformerBlock, MultiAttentionHead, AttentionHead, ResidualMLP
import engine
from engine import *
from solutions.if_then_else import if_then_else
from pygame import Vector2 as Vec2
from pygame import Rect
from .utils import *

prompt = "0010"
task = Task.from_texts([
    "0000",
    "0011",
    "0100",
    "0111",
    "1000",
    "1010",
    "1101",
    "1111",
])

model = if_then_else()
model = torch.load("./models/ITE-25k-epochs")


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


model = to_new_implem(model)


def draw(gfx: GFX):
    # draw_matrix(gfx, model.blocks[0].attention.weight)
    transformer_outline(gfx, model)


class MyState(engine.State):
    BG_COLOR = "#1a1a1a"

    def __init__(self):
        super().__init__()

        self.scale = 1
        self.translation = Vec2(0, 0)
        self.dragging = False

    def draw(self, gfx: "GFX"):
        gfx.__class__ = GFX
        gfx.screen_scale = self.scale
        gfx.translation = self.translation

        super().draw(gfx)
        draw(gfx)
        # gfx.blit(text(prompt, 100, "red"), topleft=(0, 0))

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEWHEEL and event.y and self.scale + event.y * 0.1 > 0:
                mouse_pos = pygame.mouse.get_pos()
                world_before = (Vec2(mouse_pos) -
                                self.translation) / self.scale
                self.scale *= 1 + event.y * 0.1
                self.scale = clamp(self.scale, 0.1, 10)
                world_after = (Vec2(mouse_pos) - self.translation) / self.scale
                self.translation += (world_after - world_before) * self.scale
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                self.translation += event.rel

        super().handle_events(events)