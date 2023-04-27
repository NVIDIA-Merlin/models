from typing import Dict, Final, Optional, Tuple

import torch
from torch import nn

from merlin.models.torch.data import TabularBatch
from merlin.models.torch.utils import module_utils
from merlin.models.utils.registry import Registry

registry: Registry = Registry.class_registry("torch.modules")


class BlockMixin:
    def block_init(
        self,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
    ):
        self.pre = _ModuleWrapper(pre) if pre else None
        self.post = _ModuleWrapper(post) if post else None

    def process_batch(self, batch: Optional[TabularBatch]) -> TabularBatch:
        return TabularBatch({}) if batch is None else batch

    def block_prepare(self, inputs, batch: Optional[TabularBatch] = None):
        return self.block_prepare_tensor(inputs, batch=batch)

    def block_prepare_tensor(self, inputs, batch: Optional[TabularBatch] = None):
        if self.pre is not None:
            return self.pre(inputs, batch=batch)

        return inputs

    def block_prepare_dict(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ):
        if self.pre is not None:
            return self.pre(inputs, batch=batch)

        return inputs

    def block_prepare_tuple(
        self, inputs: Tuple[torch.Tensor], batch: Optional[TabularBatch] = None
    ):
        if self.pre is not None:
            return self.pre(inputs, batch=batch)

        return inputs

    def block_prepare_batch(self, inputs: TabularBatch, batch: Optional[TabularBatch] = None):
        if self.pre is not None:
            return self.pre(inputs, batch=batch)

        return inputs

    def block_finalize(self, inputs, batch: Optional[TabularBatch] = None):
        if self.post is not None:
            return self.post(inputs, batch=batch)

        return inputs

    def block_finalize_batch(self, inputs: TabularBatch, batch: Optional[TabularBatch] = None):
        if self.post is not None:
            return self.post(inputs, batch=batch)

        return inputs

    @classmethod
    def from_registry(cls, name):
        if isinstance(name, str):
            if name not in registry:
                raise ValueError(f"Block {name} not found in registry")
            return registry.parse(name)

        raise ValueError(f"Block {name} is not a string")


class TabularBlockMixin(BlockMixin):
    def block_init(
        self,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
    ):
        self.pre = _TabularModuleWrapper(pre) if pre else None
        self.post = _TabularModuleWrapper(post) if post else None
        self.agg = _AggModuleWrapper(agg) if agg else None

    def block_prepare(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        return self.block_prepare_dict(inputs, batch=batch)

    def block_finalize(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        if self.agg is not None:
            return self._block_finalize_with_agg(inputs, batch=batch)

        return self._block_finalize_without_agg(inputs, batch=batch)

    def _block_finalize_with_agg(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ) -> torch.Tensor:
        x: Dict[str, torch.Tensor] = inputs
        if self.post is not None:
            x = self.post(x, batch=batch)

        return self.agg(x, batch=batch)

    def _block_finalize_without_agg(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ) -> Dict[str, torch.Tensor]:
        if self.post is not None:
            return self.post(inputs, batch=batch)

        return inputs

    def is_selectable(self) -> bool:
        return hasattr(self, "select_by_tag") and hasattr(self, "select_by_name")


class Block(nn.Module, BlockMixin):
    # TODO: How can we add things like append/append_with_shortcut/etc. ?

    def __init__(
        self,
        module: nn.Module = nn.Identity(),
        *,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.block_init(pre=pre, post=post)
        self.module = _ModuleWrapper(module)

        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def forward(self, inputs, batch: Optional[TabularBatch] = None):
        _batch: TabularBatch = TabularBatch({}) if batch is None else batch

        inputs = self.block_prepare(inputs, batch=_batch)
        outputs = self.forward_module(inputs, batch=_batch)

        if torch.jit.isinstance(outputs, TabularBatch):
            outputs = self.block_finalize_batch(outputs, batch=_batch)
        else:
            outputs = self.block_finalize(outputs, batch=_batch)

        return outputs

    def forward_module(self, inputs, batch: Optional[TabularBatch] = None):
        return self.module(inputs, batch=batch)

    def _get_name(self) -> str:
        if hasattr(self.module, "_get_name"):
            module_name = self.module._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class TabularIdentity(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return x


class TabularBlock(nn.Module, TabularBlockMixin):
    def __init__(
        self,
        module: nn.Module = TabularIdentity(),
        *,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
    ):
        super().__init__()

        # TODO: if tuple/Sequential is passed in and last element
        #   outputs Tensor, move it to post

        self.block_init(pre=pre, post=post, agg=agg)
        self.module = _TabularModuleWrapper(module)

        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def forward(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        inputs = self.block_prepare(inputs, batch=batch)
        outputs = self.forward_module(inputs, batch=batch)

        if torch.jit.isinstance(outputs, TabularBatch):
            outputs = self.block_finalize_batch(outputs, batch=batch)
        else:
            outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def forward_module(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        return self.module(inputs, batch=batch)

    def _get_name(self) -> str:
        if hasattr(self.module, "_get_name"):
            module_name = self.module._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class _ModuleWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    has_list: Final[bool]

    def __init__(self, to_wrap: nn.Module):
        super().__init__()

        self.to_wrap_name: str = to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        if type(to_wrap) == nn.Sequential:
            self.to_call = nn.ModuleList([_ModuleWrapper(m) for m in to_wrap])
            self.has_list = True
        else:
            self.to_call = to_wrap
            self.has_list = False
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)

    def forward(self, inputs, batch: Optional[TabularBatch] = None):
        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=batch)

            return x

        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    return self.to_call(
                        inputs, batch=batch if batch is not None else TabularBatch({})
                    )

        return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _TabularModuleWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    has_list: Final[bool]

    def __init__(self, to_wrap: nn.Module):
        super().__init__()

        self.to_wrap_name: str = to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        if type(to_wrap) == nn.Sequential or isinstance(to_wrap, tuple):
            self.to_call = nn.ModuleList([_TabularModuleWrapper(m) for m in to_wrap])
            self.has_list = True
        else:
            self.to_call = to_wrap
            self.has_list = False
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)
        # self.needs_batch = module_utils.has_batch_arg(to_wrap)

    def forward(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ) -> Dict[str, torch.Tensor]:
        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=batch)

            return x

        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    return self.to_call(
                        inputs, batch=batch if batch is not None else TabularBatch({})
                    )

            return self.to_call(inputs, batch=batch)

        return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _AggModuleWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    has_list: Final[bool]

    def __init__(self, to_wrap: nn.Module):
        super().__init__()

        self.to_wrap_name: str = to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        if type(to_wrap) == nn.Sequential:
            _wrapped = []
            for i, module in enumerate(to_wrap):
                if i == 0:
                    _wrapped.append(_AggModuleWrapper(module))
                else:
                    _wrapped.append(_ModuleWrapper(module))

            self.to_call = nn.ModuleList(_wrapped)
            self.has_list = True
        else:
            self.to_call = to_wrap
            self.has_list = False
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)

    def forward(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ) -> torch.Tensor:
        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=batch)

            return x

        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    return self.to_call(
                        inputs, batch=batch if batch is not None else TabularBatch({})
                    )

        return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name
