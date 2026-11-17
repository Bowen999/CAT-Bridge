"""Lazy imports: the heavy libraries (matplotlib, seaborn, networkx, scikit-learn, scipy.stats)
are only imported when a function that needs them is called, so ``import catbridge`` stays fast."""

import importlib


class LazyModule:
    """Stand-in for a module; the module is imported on first attribute access."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        return getattr(importlib.import_module(self._name), attr)

    def __repr__(self):
        return f"<lazy module {self._name!r}>"


def lazy_callable(module, attr):
    """Stand-in for ``from module import attr`` where attr is a class or function that is only
    called; the import happens at the first call."""
    def call(*args, **kwargs):
        return getattr(importlib.import_module(module), attr)(*args, **kwargs)
    call.__name__ = attr
    call.__qualname__ = attr
    call.__doc__ = f"Lazy wrapper of {module}.{attr}."
    return call
