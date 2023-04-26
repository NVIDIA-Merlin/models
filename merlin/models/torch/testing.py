import inspect
from typing import Dict, Final, List, Optional, Tuple, Union

import torch
from torch import nn

from merlin.models.torch.utils import padding_utils
from merlin.schema import ColumnSchema, Schema


def has_batch_arg(module: nn.Module) -> bool:
    if isinstance(module, torch.jit.ScriptModule):
        # Retrieve the schema of the forward method in the TorchScript module
        forward_schema = module.schema("forward")
        forward_signature = forward_schema.signature()
        num_args = len(forward_signature.arguments)
        has_batch_arg = any(arg.name == "batch" for arg in forward_signature.arguments)
    else:
        forward_signature = inspect.signature(module.forward)
        num_args = len(forward_signature.parameters)
        has_batch_arg = "batch" in forward_signature.parameters

    if has_batch_arg and num_args > 1:
        return True

    return False


@torch.jit.script
class TabularSequence:
    """
    A PyTorch scriptable class representing a sequence of tabular data.

    Attributes:
        lengths (Dict[str, torch.Tensor]): A dictionary mapping the feature names to their
            corresponding sequence lengths.
        masks (Dict[str, torch.Tensor]): A dictionary mapping the feature names to their
            corresponding masks. Default is an empty dictionary.

    Examples:
        >>> lengths = {'feature1': torch.tensor([4, 5]), 'feature2': torch.tensor([3, 7])}
        >>> masks = {'feature1': torch.tensor([[1, 0], [1, 1]]), 'feature2': torch.tensor([[1, 1], [1, 0]])}    # noqa: E501
        >>> tab_seq = TabularSequence(lengths, masks)
    """

    def __init__(
        self,
        lengths: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.lengths: Dict[str, torch.Tensor] = lengths
        _masks = {}
        if masks is not None:
            _masks = masks
        self.masks: Dict[str, torch.Tensor] = _masks

    def __contains__(self, name: str) -> bool:
        return name in self.lengths


@torch.jit.script
class TabularBatch:
    """
    A PyTorch scriptable class representing a batch of tabular data.

    Attributes:
        features (Dict[str, torch.Tensor]): A dictionary mapping feature names to their
            corresponding feature values.
        targets (Dict[str, torch.Tensor]): A dictionary mapping target names to their
            corresponding target values. Default is an empty dictionary.
        sequences (Optional[TabularSequence]): An optional instance of the TabularSequence class
            representing sequence lengths and masks for the batch.

    Examples:
        >>> features = {'feature1': torch.tensor([1, 2]), 'feature2': torch.tensor([3, 4])}
        >>> targets = {'target1': torch.tensor([0, 1])}
        >>> tab_batch = TabularBatch(features, targets)
    """

    def __init__(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        sequences: Optional[TabularSequence] = None,
    ):
        self.features: Dict[str, torch.Tensor] = features
        if targets is None:
            _targets = {}
        else:
            _targets = targets
        self.targets: Dict[str, torch.Tensor] = _targets
        self.sequences: Optional[TabularSequence] = sequences

    def replace(
        self,
        features: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        sequences: Optional[TabularSequence] = None,
    ) -> "TabularBatch":
        return TabularBatch(
            features=features if features is not None else self.features,
            targets=targets if targets is not None else self.targets,
            sequences=sequences if sequences is not None else self.sequences,
        )

    def __bool__(self) -> bool:
        return bool(self.features)


class _ModuleWrapper(nn.Module):
    needs_batch: Final[bool]
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
        self.needs_batch = has_batch_arg(to_wrap)

    def forward(self, inputs, batch: Optional[TabularBatch] = None):
        _batch: TabularBatch = TabularBatch({}) if batch is None else batch

        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=_batch)

            return x

        if self.needs_batch:
            return self.to_call(inputs, batch=_batch)
        else:
            return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _TabularModuleWrapper(nn.Module):
    needs_batch: Final[bool]
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
        self.needs_batch = has_batch_arg(to_wrap)

    def forward(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        _batch: TabularBatch = TabularBatch({}) if batch is None else batch

        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=_batch)

            return x

        if self.needs_batch:
            return self.to_call(inputs, batch=_batch)
        else:
            return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class _AggModuleWrapper(nn.Module):
    needs_batch: Final[bool]
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
        self.needs_batch = has_batch_arg(to_wrap)

    def forward(
        self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None
    ) -> torch.Tensor:
        _batch: TabularBatch = TabularBatch({}) if batch is None else batch

        if self.has_list:
            x = inputs
            for m in self.to_call:
                x = m(x, batch=_batch)

            return x

        if self.needs_batch:
            return self.to_call(inputs, batch=_batch)
        else:
            return self.to_call(inputs)

    def __repr__(self):
        return self.to_wrap_repr

    def _get_name(self) -> str:
        return self.to_wrap_name


class BlockMixin:
    def register_block_hooks(
        self,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
    ):
        self.pre = _ModuleWrapper(pre) if pre else None
        self.post = _ModuleWrapper(post) if post else None

    def block_prepare(self, inputs, batch: Optional[TabularBatch] = None):
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


class TabularBlockMixin:
    def register_block_hooks(
        self,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
    ):
        self.pre = _TabularModuleWrapper(pre) if pre else None
        self.post = _TabularModuleWrapper(post) if post else None
        self.agg = _AggModuleWrapper(agg) if agg else None

    def block_prepare(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        if self.pre is not None:
            return self.pre(inputs, batch=batch)

        return inputs

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


class Block(nn.Module, BlockMixin):
    def __init__(
        self,
        module: nn.Module,
        *,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.register_block_hooks(pre=pre, post=post)
        self.module = _ModuleWrapper(module)

        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def forward(self, inputs, batch: Optional[TabularBatch] = None):
        inputs = self.block_prepare(inputs, batch=batch)
        outputs = self.module(inputs, batch=batch)

        if torch.jit.isinstance(outputs, TabularBatch):
            outputs = self.block_finalize_batch(outputs, batch=batch)
        else:
            outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def _get_name(self) -> str:
        if hasattr(self.module, "_get_name"):
            module_name = self.module._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class TabularBlock(nn.Module, TabularBlockMixin):
    def __init__(
        self,
        module: nn.Module,
        *,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.register_block_hooks(pre=pre, post=post, agg=agg)
        self.module = _TabularModuleWrapper(module)

        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def forward(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        inputs = self.block_prepare(inputs, batch=batch)
        outputs = self.module(inputs, batch=batch)
        outputs = self.block_finalize(outputs, batch=batch)

        return outputs

    def _get_name(self) -> str:
        if hasattr(self.module, "_get_name"):
            module_name = self.module._get_name()

            if not module_name.endswith("Block"):
                module_name += "Block"

            return module_name

        return super()._get_name()


class Model(nn.Module, TabularBlockMixin):
    def __init__(
        self,
        module: nn.Module,
        *,
        pre: Optional[nn.Module] = None,
        post: Optional[nn.Module] = None,
        agg: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.register_block_hooks(pre=pre, post=post, agg=agg)
        self.module = _TabularModuleWrapper(module)

        if hasattr(module, "input_schema"):
            self.input_schema = module.input_schema
        if hasattr(module, "output_schema"):
            self.output_schema = module.output_schema

    def forward(self, inputs: Dict[str, torch.Tensor], batch: Optional[TabularBatch] = None):
        _batch: TabularBatch = TabularBatch({}) if batch is None else batch

        prepared = self.block_prepare(inputs, batch=batch)
        if torch.jit.isinstance(prepared, TabularBatch):
            _batch = prepared
            inputs = prepared.features
        else:
            inputs = prepared

        outputs = self.module(inputs, batch=_batch)
        outputs = self.block_finalize(outputs, batch=_batch)

        return outputs


class TabularPadding(nn.Module):
    """
    A PyTorch Module for padding tabular sequences in a batch.

    Attributes:
        schema (Schema): The schema object representing the structure of the input data.
        target (Union[ColumnSchema, Schema]): A ColumnSchema or Schema object representing
            the target columns for the padding.
        max_sequence_length (Optional[int]): The maximum sequence length for padding. If not
            specified, the maximum length in the batch will be used. Default is None.
        features (List[str]): A list of feature names from the input schema.
    """

    def __init__(
        self,
        schema: Schema,
        target: Union[ColumnSchema, Schema],
        max_sequence_length: Optional[int] = None,
    ):
        super().__init__()
        self.schema = schema
        self.target = target if isinstance(target, Schema) else Schema([target])
        self.max_sequence_length = max_sequence_length
        self.features: List[str] = self.schema.column_names

    def forward(self, batch: TabularBatch) -> TabularBatch:
        seqs, outputs = {}, {}

        for key, val in batch.features.items():
            _key: str = key
            if key.endswith("__values"):
                _key = key[: -len("__values")]
            elif key.endswith("__offsets"):
                _key = key[: -len("__offsets")]

            if _key in self.features:
                seqs[key] = val
            else:
                outputs[key] = val

        # TODO: Should we pad the targets?
        padded, lengths = padding_utils.pad_inputs(seqs, self.max_sequence_length)
        for key, val in padded.items():
            outputs[key] = val

        return TabularBatch(outputs, batch.targets, sequences=TabularSequence(lengths))
