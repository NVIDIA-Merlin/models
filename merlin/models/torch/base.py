from typing import Protocol, Union, runtime_checkable

from torch import nn
from typing_extensions import Self

from merlin.models.torch.utils.module_utils import ModulePreHook, ModulePostHook
from merlin.models.utils.registry import Registry

registry: Registry = Registry.class_registry("torch.modules")


class Block(nn.Module):
    """
    A custom PyTorch module that applies pre and post processing to the input.

    Parameters
    ----------
    pre : nn.Module, optional
        The module to apply before the main processing.
    post : nn.Module, optional
        The module to apply after the main processing.
    """

    def __init__(self, pre=None, post=None):
        super().__init__()
        # TODO: How to know whether or not to forward kwargs?
        self.pre = register_pre_hook(self, pre) if pre else None
        self.post = register_post_hook(self, post) if post else None

    @classmethod
    def from_registry(cls, name):
        if isinstance(name, str):
            if name not in registry:
                raise ValueError(f"Block {name} not found in registry")
            return registry.parse(name)

        raise ValueError(f"Block {name} is not a string")

    def forward(self, inputs):
        return inputs


@runtime_checkable
class Selectable(Protocol):
    def select_by_name(self, names):
        ...

    def select_by_tag(self, tags):
        ...


class TabularBlock(Block):
    """
    A custom PyTorch module that applies pre and post
    processing and an aggregation function to the input.

    Parameters
    ----------
    pre : nn.Module, optional
        The module to apply before the main processing.
    post : nn.Module, optional
        The module to apply after the main processing.
    aggregation : Callable, optional
        The function to apply on the output tensor.
    """

    def __init__(self, pre=None, post=None, aggregation=None):
        super().__init__(pre=pre, post=post)
        self.aggregation = register_post_hook(self, aggregation) if aggregation else None

    def forward(self, inputs):
        return inputs

    # @property
    # def is_selectable(self) -> bool:
    #     return isinstance(self, Selectable)


def register_pre_hook(
    module: nn.Module,
    to_register: Union[str, nn.Module],
    prepend: bool = False,
    with_kwargs: bool = False,
) -> nn.Module:
    """Register a pre-hook for a PyTorch module.

    Args:
        module (nn.Module): The module to register the pre-hook for.
        to_register (Union[str, nn.Module]): The pre-hook to register.
            It can be a string (name of the block) or an instance of nn.Module.
        prepend (bool, optional): If True, prepend the pre-hook to the existing pre-hooks.
            Defaults to False.
        with_kwargs (bool, optional): If True, the pre-hook will receive kwargs. Defaults to False.

    Returns:
        nn.Module: The registered pre-hook.
    """
    pre = Block.from_registry(to_register) if isinstance(to_register, str) else to_register

    module.register_forward_pre_hook(ModulePreHook(pre), prepend=prepend, with_kwargs=with_kwargs)

    return pre


def register_post_hook(
    module: nn.Module,
    to_register: Union[str, nn.Module],
    prepend: bool = False,
    with_kwargs: bool = False,
) -> nn.Module:
    """Register a post-hook for a PyTorch module.

    Args:
        module (nn.Module): The module to register the post-hook for.
        to_register (Union[str, nn.Module]): The post-hook to register.
            It can be a string (name of the block) or an instance of nn.Module.
        prepend (bool, optional): If True, prepend the post-hook to the existing post-hooks.
            Defaults to False.
        with_kwargs (bool, optional): If True, the post-hook will receive kwargs.
            Defaults to False.

    Returns:
        nn.Module: The registered post-hook.
    """
    post = Block.from_registry(to_register) if isinstance(to_register, str) else to_register

    module.register_forward_hook(ModulePostHook(post), prepend=prepend, with_kwargs=with_kwargs)

    return post
