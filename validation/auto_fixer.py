"""
Compatibility entrypoint for output auto-fixing.

The generated implementation is timestamped, while the package initializer
imports `validation.auto_fixer`.
"""
from .auto_fixer_20260428_231043 import AutoFixer

__all__ = ["AutoFixer"]
