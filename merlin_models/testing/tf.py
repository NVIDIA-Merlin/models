# flake8: noqa

import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

import merlin_models.tf as ml

__all__ = ["tf", "ml"]
