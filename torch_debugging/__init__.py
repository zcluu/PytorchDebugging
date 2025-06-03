import importlib


def __getattr__(name):
    if name in ["ops"]:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")
