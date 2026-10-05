from fire import Fire

from cli import App

try:
    Fire(App, name="RAGAgainstTheMachine")
except KeyboardInterrupt:
    pass
except Exception as e:
    print("Error:", e)
