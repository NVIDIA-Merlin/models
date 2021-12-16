# flake8: noqa

import pytest

pytest.importorskip("torch")

import torch

import merlin_models.torch as ml

__all__ = ["torch", "ml"]
