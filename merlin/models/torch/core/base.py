from typing import Union

from torch import nn

from merlin.models.utils.registry import Registry

registry: Registry = Registry.class_registry("torch.modules")


class _ModuleHook(nn.Module):
    """A helper module to be able to execute Module as a (pre) forward-hook."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, parent, inputs, outputs=None):
        """
        Forward pass for the _ModuleHook.

        This function is called when the hook is executed during the
        forward pass of the parent module.

        Args:
            parent (nn.Module): The parent module for which the hook is registered.
            inputs: The inputs to the parent module.
            outputs (optional): The outputs of the parent module (only provided for post-hooks).

        Returns:
            The result of executing the hook module with the given inputs or outputs.
        """
        del parent

        x = inputs if outputs is None else outputs
        if isinstance(x, tuple):
            return self.module(*x)

        return self.module(x)


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


class NoOp(nn.Module):
    """A module that simply passes the input through unchanged.

    This is useful as a placeholder module or when you need a module that
    does not modify the input in any way, for instance a short-cut connection.
    """

    def forward(self, inputs):
        """
        Forward pass for the NoOp module.

        Args:
            inputs: Input tensor.

        Returns:
            The input tensor unchanged.
        """
        return inputs


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

    module.register_forward_pre_hook(_ModuleHook(pre), prepend=prepend, with_kwargs=with_kwargs)

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

    module.register_forward_hook(_ModuleHook(post), prepend=prepend, with_kwargs=with_kwargs)

    return post
