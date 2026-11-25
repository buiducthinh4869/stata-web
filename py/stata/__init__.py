"""Stata emulator for teaching causal inference (runs in Pyodide)."""
from .interp import Interp, load_commands

load_commands()

__all__ = ['Interp']
