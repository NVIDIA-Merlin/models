from typing import Any

try:
    import nvtabular as nvt

    ColumnGroup = nvt.ColumnGroup
except ImportError:
    ColumnGroup = Any

__all__ = ["ColumnGroup"]
