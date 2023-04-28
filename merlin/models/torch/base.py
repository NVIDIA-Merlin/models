from typing import Dict, Final, Optional, Tuple, Union

import torch
from torch import nn

from merlin.models.torch.container import Parallel, WithShortcut, _WrappedModuleList
from merlin.models.torch.data import Batch
from merlin.models.torch.utils import module_utils
from merlin.models.utils.registry import Registry

registry: Registry = Registry.class_registry("torch.modules")


class BlockMixin:
    def block_init(
        self,
        module: nn.Module,
        *extra: nn.Module,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ):
        _pre = []
        if pre:
            _pre = [pre] if not isinstance(pre, (list, tuple)) else pre
        self.pre = _WrappedModuleList(_pre)

        _post = []
        if post:
            _post = [post] if not isinstance(post, (list, tuple)) else post
        self.post = _WrappedModuleList(_post)

        self.block = _WrappedModuleList([module, *extra])
        self._name = name

        # TODO: Is this needed?
        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def set_agg(self, agg: Union[str, nn.Module]):
        raise NotImplementedError()

    def append(self, module: nn.Module):
        self.block.append(module)

        return self

    def append_branch(
        self,
        *branches: Union[nn.Module, Dict[str, nn.Module]],
        aggregation=None,
        **kwargs,
    ):
        self.append(Parallel(*branches, **kwargs))

        if aggregation:
            self.append(aggregation)

        return self

    def process_batch(self, batch: Optional[Batch]) -> Batch:
        return Batch({}) if batch is None else batch

    def block_prepare(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        for pre in self.pre:
            inputs = pre(inputs, batch=batch)

        return inputs

    # def block_prepare(self, inputs, batch: Optional[TabularBatch] = None):
    #     return self.block_prepare_tensor(inputs, batch=batch)

    def block_prepare_tensor(self, inputs, batch: Optional[Batch] = None):
        for pre in self.pre:
            inputs = pre(inputs, batch=batch)

        return inputs

    def block_prepare_dict(self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None):
        for pre in self.pre:
            inputs = pre(inputs, batch=batch)

        return inputs

    def block_prepare_tuple(self, inputs: Tuple[torch.Tensor], batch: Optional[Batch] = None):
        for pre in self.pre:
            inputs = pre(inputs, batch=batch)

        return inputs

    def block_prepare_batch(self, inputs: Batch, batch: Optional[Batch] = None):
        for pre in self.pre:
            inputs = pre(inputs, batch=batch)

        return inputs

    def block_finalize(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        for post in self.post:
            inputs = post(inputs, batch=batch)

        return inputs

    # def block_finalize(self, inputs, batch: Optional[TabularBatch] = None):
    #     for post in self.post:
    #         inputs = post(inputs, batch=batch)

    #     return inputs

    def block_finalize_batch(self, inputs: Batch, batch: Optional[Batch] = None):
        for post in self.post:
            inputs = post(inputs, batch=batch)

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

    def block_prepare(self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None):
        return self.block_prepare_dict(inputs, batch=batch)

    def block_finalize(self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None):
        if self.agg is not None:
            return self._block_finalize_with_agg(inputs, batch=batch)

        return self._block_finalize_without_agg(inputs, batch=batch)

    def _block_finalize_with_agg(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
    ) -> torch.Tensor:
        x: Dict[str, torch.Tensor] = inputs
        if self.post is not None:
            x = self.post(x, batch=batch)

        return self.agg(x, batch=batch)

    def _block_finalize_without_agg(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
    ) -> Dict[str, torch.Tensor]:
        if self.post is not None:
            return self.post(inputs, batch=batch)

        return inputs

    def is_selectable(self) -> bool:
        return hasattr(self, "select_by_tag") and hasattr(self, "select_by_name")


class Block(nn.Module, BlockMixin):
    def __init__(
        self,
        module: nn.Module = nn.Identity(),
        *extra: nn.Module,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.block_init(module, *extra, pre=pre, post=post, name=name)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        inputs = self.block_prepare(inputs, batch=batch)
        outputs = self.call(inputs, batch=batch)

        if torch.jit.isinstance(outputs, Batch):
            outputs = self.block_finalize_batch(outputs, batch=batch)
        else:
            outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def call(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        return self.call_modules(inputs, batch=batch)

    def call_modules(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        for module in self.block:
            inputs = module(inputs, batch=batch)

        return inputs

    def _get_name(self) -> str:
        if self._name is not None:
            return self._name

        if len(self.block) == 1:
            module_name = self.block[0]._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class TabularIdentity(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return x


class ParallelBlock(Block):
    def __init__(
        self,
        *inputs: Union[nn.Module, Dict[str, nn.Module]],
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
        strict: bool = True,
    ):
        module = Parallel(*inputs, strict=strict)

        # TODO: Handle agg

        super().__init__(module, pre=pre, post=post)

    def append(self, module: Optional[nn.Module] = None, **branches):
        if module is None:
            if len(branches) == 0:
                raise ValueError("Module or branches must be provided")

            if not all(isinstance(b, nn.Module) for b in branches.values()):
                raise ValueError("All branches must be nn.Module")

            if not all(k in self[-1].branches for k in branches.keys()):
                raise ValueError("All branches must be in previous module")

            module = Parallel(**branches)
        else:
            if not hasattr(module, "branches"):
                raise ValueError("Module must have branches attribute")

        return super().append(module)


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

    def forward(self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None):
        inputs = self.block_prepare(inputs, batch=batch)
        outputs = self.forward_module(inputs, batch=batch)

        if torch.jit.isinstance(outputs, Batch):
            outputs = self.block_finalize_batch(outputs, batch=batch)
        else:
            outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def forward_module(self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None):
        return self.block(inputs, batch=batch)

    def _get_name(self) -> str:
        if hasattr(self.module, "_get_name"):
            module_name = self.module._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class _Wrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]

    def __init__(self, to_wrap: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.to_wrap = to_wrap
        self.to_wrap_name: str = name or to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)

    def forward(self, inputs, batch: Optional[Batch] = None):
        if self.accepts_batch:
            if self.requires_batch:
                if batch is None:
                    raise RuntimeError("batch is required for this module")
                else:
                    # else-clause is needed to make torchscript happy
                    _batch = batch if batch is not None else Batch({})

                    return self.to_wrap(inputs, batch=_batch)

        return self.to_wrap(inputs)

    def unwrap(self) -> nn.Module:
        return self.to_wrap

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _ModuleWrapper(nn.Module):
    accepts_batch: Final[bool]
    requires_batch: Final[bool]
    has_list: Final[bool]

    def __init__(self, to_wrap: nn.Module, *extra: nn.Module):
        super().__init__()

        _wrapped = _ModuleWrapper(extra) if len(extra) > 0 else None

        self.to_wrap_name: str = to_wrap._get_name()
        self.to_wrap_repr: str = repr(to_wrap)
        if type(to_wrap) == nn.Sequential or isinstance(to_wrap, tuple):
            list = [_ModuleWrapper(m) for m in to_wrap]
            if _wrapped is not None:
                list.append(_wrapped)
            self.to_call = nn.ModuleList(list)
            self.has_list = True
        elif _wrapped is not None:
            self.to_call = nn.ModuleList([_ModuleWrapper(to_wrap), _wrapped])
            self.has_list = True
        else:
            self.to_call = to_wrap
            self.has_list = False
        self.accepts_batch, self.requires_batch = module_utils.check_batch_arg(to_wrap)

    def forward(self, inputs, batch: Optional[Batch] = None):
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
                    return self.to_call(inputs, batch=batch if batch is not None else Batch({}))

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
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
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
                    return self.to_call(inputs, batch=batch if batch is not None else Batch({}))

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
        self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
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
                    return self.to_call(inputs, batch=batch if batch is not None else Batch({}))

        return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name
