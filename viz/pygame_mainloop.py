import threading
import traceback
import subprocess
from time import sleep
import io


def pygame_mainloop_thread_start(globals_: dict):
    pyapp = None

    def run_app():
        import pygame
        from viz import engine
        from viz.main import JupyterState
        nonlocal pyapp
        while True:
            pyapp = engine.App(JupyterState, engine.FreeScreen((1000, 500)),
                               globals_)
            pygame.mouse.set_visible(True)
            try:
                pyapp.run()
            except Exception as error:
                stream = io.StringIO()
                traceback.print_exc(file=stream)
                print(stream.getvalue())
                subprocess.run(
                    ["notify-send", "Pygame app crashed!",
                     stream.getvalue()])
                print("App crashed!!!")
            sleep(1)

    thread = threading.Thread(target=run_app)
    thread.start()

    return lambda: pyapp.quit()
