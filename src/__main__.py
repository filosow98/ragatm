from cli import App
from fire import Fire


try:
    Fire(App, name="RAGAgainstTheMachine")
except KeyboardInterrupt as _:
    pass
except Exception as e:
    print(e)
