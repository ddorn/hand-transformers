from contextlib import contextmanager

import pygame
import pygame.gfxdraw
from pygame import Vector2 as Vec2, Rect

from viz.engine.utils import chrange

from .constants import *
from .assets import font, text

__all__ = ["GFX"]


class GFX:

    def __init__(self, surf: pygame.Surface):
        self.surf = surf
        self.screen_scale = 1
        """How many pixel is one world unit."""
        self.translation = pygame.Vector2()
        """Translate all draw/blit by this amount."""

    def to_screen(self, pos):
        """Convert a position or rect in the screen to world coordinates."""
        if len(pos) == 2:
            return Vec2(pos) * self.screen_scale + self.translation
        elif len(pos) == 4:
            return pygame.Rect(
                pos[0] * self.screen_scale + self.translation.x,
                pos[1] * self.screen_scale + self.translation.y,
                pos[2] * self.screen_scale,
                pos[3] * self.screen_scale,
            )
        else:
            raise ValueError("Invalid position")

    def to_world(self, screen_pos):
        """Convert a position or rect in the screen to world coordinates."""
        if len(screen_pos) == 2:
            return (Vec2(screen_pos) - self.translation) / self.screen_scale
        elif len(screen_pos) == 4:
            return pygame.Rect(
                (screen_pos[0] - self.translation.x) / self.screen_scale,
                (screen_pos[1] - self.translation.y) / self.screen_scale,
                screen_pos[2] / self.screen_scale,
                screen_pos[3] / self.screen_scale,
            )
        else:
            raise ValueError("Invalid position")

    # Positions / size conversion functions
    '''
    def to_ui(self, pos):
        """Convert a position in the screen to ui coordinates."""
        return pygame.Vector2(pos) / self.ui_scale

    def scale_ui_pos(self, x, y):
        return (int(x * self.surf.get_width()), int(y * self.surf.get_height()))

    def scale_world_size(self, w, h):
        return (int(w * self.world_scale), int(h * self.world_scale))

    def scale_world_pos(self, x, y):
        return (
            int((x - self.world_center.x) * self.world_scale)
            + self.surf.get_width() // 2,
            int((y - self.world_center.y) * self.world_scale)
            + self.surf.get_height() // 2,
        )

    # Surface related functions

    @lru_cache(maxsize=2000)
    def scale_surf(self, surf: pygame.Surface, factor):
        if factor == 1:
            return surf
        if isinstance(factor, (tuple, pygame.Vector2)):
            size = (int(factor[0]), int(factor[1]))
        else:
            size = (int(surf.get_width() * factor), int(surf.get_height() * factor))
        return pygame.transform.scale(surf, size)

    def ui_blit(self, surf: pygame.Surface, **anchor):
        assert len(anchor) == 1
        anchor, value = anchor.popitem()

        s = self.scale_surf(surf, self.ui_scale)
        r = s.get_rect(**{anchor: self.scale_ui_pos(*value)})
        self.surf.blit(s, r)

    def world_blit(self, surf, pos, size, anchor="topleft"):
        s = self.scale_surf(surf, vec2int(size * self.world_scale))
        r = s.get_rect(**{anchor: pos * self.world_scale})
        r.topleft -= self.world_center
        self.surf.blit(s, r)
    '''

    def blit(self, surf, **anchor):
        """Blit a surface directly on the underlying surface, coordinates are in pixels."""

        r = surf.get_rect(**anchor)
        self.surf.blit(surf, self.to_screen(r.topleft))

        return r

    # Draw functions

    def rect(self, x, y, w, h, color, width=0, anchor: str = None, abs_position=False):
        """Draw a rectangle in rect coordinates."""
        r = pygame.Rect(x, y, w, h)

        if anchor:
            setattr(r, anchor, (x, y))
        if not abs_position:
            r = self.to_screen(r)
        pygame.draw.rect(self.surf, color, r, width)

    def box(self, rect, color, abs_position=False):
        color = pygame.Color(color)
        if not abs_position:
            rect = self.to_screen(rect)
        pygame.gfxdraw.box(self.surf, rect, color)

    def circle(self, pos, radius, color, width=0):
        pygame.draw.circle(self.surf, color, self.to_screen(pos),
                           radius * self.screen_scale, width)

    def text(self,
             txt,
             size=DEFAULT_TEXT_SIZE,
             color=WHITE,
             abs_position=False,
             **anchor) -> pygame.Rect:
        assert len(anchor) == 1
        txt = str(txt)
        if not txt:
            return pygame.Rect(0, 0, 0, 0)
        s = text(txt, round(size * self.screen_scale), color)
        r = pygame.Rect(0, 0, *font(round(size)).size(txt))
        anchor, pos = anchor.popitem()
        setattr(r, anchor, pos)
        if not abs_position:
            self.surf.blit(s, self.to_screen(r))
        else:
            self.surf.blit(s, r)
        return r

    def texts(self, *texts, color=WHITE, **anchor) -> Rect:
        assert len(anchor) == 1
        anchor, pos = anchor.popitem()
        surfs = []
        sizes = []
        for txt in texts:
            surfs.append(
                text(txt, round(DEFAULT_TEXT_SIZE * self.screen_scale), color))
            sizes.append(font(DEFAULT_TEXT_SIZE).size(txt))

        world_rect = Rect(0, 0, max(s[0] for s in sizes),
                          sum(s[1] for s in sizes))
        setattr(world_rect, anchor, pos)

        y = world_rect.top
        for surf, size in zip(surfs, sizes):
            line_rect = Rect(world_rect.left, y, world_rect.width, size[1])
            if "left" in anchor:
                x = line_rect.left
            elif "right" in anchor:
                x = line_rect.right - size[0]
            else:
                x = line_rect.centerx - size[0] // 2
            self.surf.blit(surf, self.to_screen((x, y)))
            y += size[1]

        return world_rect

    def plot(self,
             points,
             color='white',
             title='',
             size=100,
             axis=False,
             **anchor):
        plot_rect = Rect(0, 0, size, size)
        full_rect = plot_rect.copy()
        if title:
            y = font(DEFAULT_TEXT_SIZE).size(title)[1]
            full_rect.height += y
        if axis:
            w = font(DEFAULT_TEXT_SIZE).size('0.0')[0]
            full_rect.width += w
        anchor, pos = anchor.popitem()
        setattr(full_rect, anchor, pos)
        plot_rect.bottomright = full_rect.bottomright

        if title:
            self.text(title, midtop=full_rect.midtop)
        self.rect(*plot_rect, 'white', 1)

        if len(points) < 2:
            return full_rect

        maxi = max(points)
        mini = min(points)
        if maxi == mini:
            maxi += 1
            mini -= 1
        ys = [
            chrange(p, (mini, maxi), (plot_rect.bottom, plot_rect.bottom - size))
            for p in points
        ]
        xs = [
            chrange(i, (0, len(points)), (plot_rect.left + 1, plot_rect.right - 1))
            for i in range(len(points))
        ]

        pygame.draw.lines(self.surf, color, False,
                          [self.to_screen(p) for p in zip(xs, ys)], 1)

        if axis:
            self.text(f"{mini:.1f}", midright=plot_rect.bottomleft)
            self.text(f"{maxi:.1f}", midright=plot_rect.topleft)
            if len(points) > 1:
                curr = points[-1]
                self.text(f"{curr:.1f}", midleft=(plot_rect.right, float(ys[-1])))

        return full_rect

    def grid(self, pos, blocks, steps, color=(255, 255, 255, 100)):
        """
        Draw a grid.

        Args:
            surf: The surface on which to draw
            pos: position of the topleft corner
            blocks (Tuple[int, int]): Number of columns and rows (width, height)
            steps: size of each square block, in world coordinates
            color: Color of the grid. Supports alpha.
        """

        color = pygame.Color(color)
        left, top = self.to_screen(pos)
        bottom = round(top + blocks[0] * steps * self.screen_scale)
        right = round(left + blocks[1] * steps * self.screen_scale)
        for x in range(blocks[0] + 1):
            x = round(left + x * steps * self.screen_scale)
            pygame.gfxdraw.line(self.surf, x, round(top), x, bottom, color)
        for y in range(blocks[0] + 1):
            y = round(top + y * steps * self.screen_scale)
            pygame.gfxdraw.line(self.surf, round(left), y, right, y, color)

    def fill(self, color):
        self.surf.fill(color)

    def scroll(self, dx, dy):
        self.surf.scroll(dx, dy)

    @contextmanager
    def focus(self, rect):
        """Set the draw rectangle with clip, and translate all draw calls
        so that (0, 0) is the topleft of the given rectangle.
        """

        rect = pygame.Rect(rect)

        previous_clip = self.surf.get_clip()
        self.surf.set_clip(rect)
        self.translation = pygame.Vector2(rect.topleft)
        yield
        self.surf.set_clip(previous_clip)
        if previous_clip:
            self.translation = pygame.Vector2(previous_clip.topleft)
