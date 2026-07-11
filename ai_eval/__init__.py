"""ai-evals — zero-config, repo-aware, model-agnostic AI evaluation.

See `.kilo/plans/ai-evals-cli-and-system-design.md` for the full system design.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dot-ai-evals")
except PackageNotFoundError:  # editable/source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
