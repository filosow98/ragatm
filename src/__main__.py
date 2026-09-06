from cli import App
from fire import Fire

try:
    Fire(App)
except KeyboardInterrupt as _:
    pass
except Exception as e:
    print(e)
