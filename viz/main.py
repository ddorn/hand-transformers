from collections import defaultdict
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import subprocess
import sys
from time import ctime, sleep, time
from typing import Type
import pygame
import click
import watchfiles
import threading
import io
import traceback

from . import engine

from pygame import Vector2 as Vec2

PATH = Path(__file__).parent


def load_script(path: Path):
    """Import a source file as a module."""
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AutoReloadApp(engine.App):
    NAME = 'Pygame window'
    MOUSE_VISIBLE = True

    def __init__(self, state_path: Path, reload_dir: Path,
                 resizing: engine.Screen):
        self.state_path = state_path
        self.reload_dir = reload_dir
        self.last_loaded = 0
        self.crashed = None
        super().__init__(engine.State, resizing)
        with self.catch_crash():
            self.reload(recurse=False)
        self.stop_event = threading.Event()

        def thread():
            for change in watchfiles.watch(self.reload_dir,
                                           stop_event=self.stop_event):
                print(change)
                self.reload()

        self.watch_thread = threading.Thread(target=thread)
        self.watch_thread.start()

    def events(self):
        events = super().events()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reload()
        return events

    def handle_events(self, events):
        if not self.crashed:
            with self.catch_crash():
                self.state.handle_events(events)

    def logic(self):
        if not self.crashed:
            with self.catch_crash():
                self.state.logic()

        # if time() - self.last_loaded > 0.5 and self.state_path.stat(
        # ).st_mtime > self.last_loaded:
        #     self.reload()

    def draw(self):
        if not self.crashed:
            with self.catch_crash():
                super().draw()
        else:
            # format exception well, with traceback
            buffer = io.StringIO()
            traceback.print_exception(type(self.crashed),
                                      self.crashed,
                                      self.crashed.__traceback__,
                                      file=buffer)
            self.gfx.fill('black')
            txt = engine.wrapped_text(buffer.getvalue(),
                                      30,
                                      'red',
                                      10000,
                                      center=False)
            self.gfx.blit(txt, topleft=(0, 0))

    def reload(self, recurse=True) -> Type[engine.State]:
        print(ctime(), '(re)loading script', self.state_path)
        self.last_loaded = time()
        self.crashed = None

        with self.catch_crash():
            if recurse:
                for mod in list(sys.modules.values()):
                    if (hasattr(mod, "__file__") and mod.__file__ is not None
                            and mod.__name__ != '__main__'
                            and self.reload_dir in Path(mod.__file__).parents):
                        print('Reloading', mod.__name__)
                        importlib.reload(mod)

            module = load_script(self.state_path)

            states = {
                name: cls
                for name, cls in module.__dict__.items()
                if isinstance(cls, type) and issubclass(cls, engine.State)
                and cls is not engine.State
            }

            if len(states) == 1:
                state = next(iter(states.values()))()
                self.state.replace_state(state)
            else:
                raise ValueError('Not exactly one state found in script')

    @contextmanager
    def catch_crash(self):
        try:
            yield
        except Exception as e:
            self.crashed = e
            traceback.print_exception(type(e), e, e.__traceback__)

    def run(self):
        super().run()
        # Kill the watch thread
        self.stop_event.set()


class JupyterState(engine.State):
    FPS = 30
    BG_COLOR = '#222222'
    def __init__(self, globals_: dict):
        super().__init__()
        self.globals = globals_
        self.crashed = None
        self.last_methods = {
            'handle_events': None,
            'logic': None,
            'draw': None,
        }

        self.scale = 1
        self.translation = Vec2(0, 0)
        self.dragging = False

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEWHEEL and event.y and self.scale + event.y * 0.1 > 0:
                mouse_pos = pygame.mouse.get_pos()
                world_before = (Vec2(mouse_pos) -
                                self.translation) / self.scale
                self.scale *= 1 + event.y * 0.1
                self.scale = min(10, max(0.1, self.scale))
                world_after = (Vec2(mouse_pos) - self.translation) / self.scale
                self.translation += (world_after - world_before) * self.scale
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                self.translation += event.rel

        if not self.crashed:
            super().handle_events(events)
            with self.catch_crash():
                self.globals.get('handle_events', lambda events: None)(events)

    def logic(self):
        if not self.crashed:
            super().logic()
            with self.catch_crash():
                self.globals.get('logic', lambda: None)()
        else:
            self.timer += 1
            if self.timer % 60 == 0:
                self.crashed = None
            for method in ('handle_events', 'logic', 'draw'):
                if self.last_methods[method] != self.globals.get(method):
                    self.crashed = None
                    self.last_methods[method] = self.globals.get(method)

    def draw(self, gfx: engine.GFX):
        gfx.screen_scale = self.scale
        gfx.translation = self.translation

        if not self.crashed:
            super().draw(gfx)
            with self.catch_crash():
                try:
                    self.globals.get('draw', lambda gfx: None)(gfx)
                except Exception as e:
                    self.crashed = e
                    traceback.print_exception(type(e), e, e.__traceback__)
                # gfx.__class__ = engine.GFX
        else:
            # format exception well, with traceback
            buffer = io.StringIO()
            traceback.print_exception(type(self.crashed),
                                      self.crashed,
                                      self.crashed.__traceback__,
                                      file=buffer)
            gfx.fill('black')
            txt = engine.wrapped_text(buffer.getvalue(),
                                      30,
                                      'red',
                                      10000,
                                      center=False)
            gfx.blit(txt, topleft=(0, 0))
        gfx.text(self.timer, topleft=(0, 0))

    @contextmanager
    def catch_crash(self):
        try:
            yield
        except Exception as e:
            self.crashed = e
            traceback.print_exception(type(e), e, e.__traceback__)


@click.command()
@click.argument('script-path',
                default='viz/script.py',
                type=click.Path(exists=True, path_type=Path))
@click.option('--reload-dir',
              default=PATH.parent,
              type=click.Path(exists=True, path_type=Path, file_okay=False))
def main(script_path: Path, reload_dir: Path):

    app = AutoReloadApp(script_path, reload_dir,
                        engine.screen.FreeScreen((1400, 700)))
    app.run()


if __name__ == '__main__':
    main()