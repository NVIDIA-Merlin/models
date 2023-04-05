import inspect
import itertools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import torch
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as TorchrecEmbeddingBagCollection,
)
from torchrec.modules.embedding_modules import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingCollection as TorchrecEmbeddingCollection
from torchrec.modules.embedding_modules import EmbeddingConfig
from torchrec.modules.fused_embedding_modules import EmbeddingLocation
from torchrec.modules.fused_embedding_modules import (
    FusedEmbeddingBagCollection as TorchrecFusedEmbeddingBagCollection,
)
from torchrec.modules.fused_embedding_modules import (
    FusedEmbeddingCollection as TorchrecFusedEmbeddingCollection,
)
from typing_extensions import Self

from merlin.models.torch.base import register_post_hook, register_pre_hook
from merlin.models.torch.inputs.embedding import _get_dim
from merlin.models.torch.torchrec.transforms import ToKeyedJaggedTensor
from merlin.models.utils.schema_utils import infer_embedding_dim
from merlin.schema import ColumnSchema, Schema, Tags

Config = TypeVar("Config", bound=BaseEmbeddingConfig)


class TorchrecEmbeddingMixin:
    def parse_tables(
        self, schema_or_tables, config_cls: Type[Config] = EmbeddingConfig, **table_kwargs
    ) -> List[Config]:
        if isinstance(schema_or_tables, Schema):
            tables = create_embedding_configs(
                schema_or_tables, config_cls=config_cls, **table_kwargs
            )
            self.schema = schema_or_tables
        else:
            tables = schema_or_tables

        return tables

    def register_hooks(self, pre=None, post=None):
        self.pre = register_pre_hook(self, pre) if pre else None
        self.post = register_post_hook(self, post) if post else None
        self.to_keyed_jagged_tensor = register_pre_hook(
            self, ToKeyedJaggedTensor(self.input_schema)
        )

    @property
    def input_schema(self) -> Schema:
        if getattr(self, "schema", None):
            return self.schema

        # TODO: We could get int-domains as well from the embedding configs

        cols = [
            ColumnSchema(name=col.name, is_list=True)
            for col in itertools.chain(*self._feature_names)
        ]
        return Schema(cols)

    def select_by_name(self, names: Union[str, List[str]]) -> Self:
        raise NotImplementedError("TODO")

    def select_by_tag(self, tags: Union[Tags, List[Tags]]) -> Self:
        raise NotImplementedError("TODO")


class EmbeddingBagCollection(TorchrecEmbeddingBagCollection, TorchrecEmbeddingMixin):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B * (F * D)] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        device (Optional[torch.device]): default compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.8899, -0.1342, -1.9060, -0.0905, -0.2814, -0.9369, -0.7783],
            [ 0.0000,  0.0000,  0.0000,  0.1598,  0.0695,  1.3265, -0.1011],
            [-0.4256, -1.1846, -2.1648, -1.0893,  0.3590, -1.9784, -0.7681]],
            grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        schema_or_tables: Union[Schema, List[EmbeddingBagConfig]],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        pre=None,
        post=None,
        **table_kwargs
    ):
        super().__init__(
            self.parse_tables(schema_or_tables, config_cls=EmbeddingBagConfig, **table_kwargs),
            is_weighted=is_weighted,
            device=device,
        )
        self.register_hooks(pre=pre, post=post)


class EmbeddingCollection(TorchrecEmbeddingCollection, TorchrecEmbeddingMixin):
    """
    EmbeddingCollection represents a collection of non-pooled embeddings.

    It processes sparse data in the form of `KeyedJaggedTensor` of the form [F X B X L]
    where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (variable)

    and outputs `Dict[feature (key), JaggedTensor]`.
    Each `JaggedTensor` contains values of the form (B * L) X D
    where:

    * B: batch size
    * L: length of sparse features (jagged)
    * D: each feature's (key's) embedding dimension and lengths are of the form L

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup dict.

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([[-0.2050,  0.5478,  0.6054],
        [ 0.7352,  0.3210, -3.0399],
        [ 0.1279, -0.1756, -0.4130],
        [ 0.7519, -0.4341, -0.0499],
        [ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>)
    """

    def __init__(
        self,
        schema_or_tables: Union[Schema, List[EmbeddingConfig]],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
        pre=None,
        post=None,
        **table_kwargs
    ):
        super().__init__(
            self.parse_tables(schema_or_tables, **table_kwargs),
            need_indices=need_indices,
            device=device,
        )
        self.register_hooks(pre=pre, post=post)


class FusedEmbeddingBagCollection(TorchrecFusedEmbeddingBagCollection, TorchrecEmbeddingMixin):
    """
    FusedEmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).
    It utilizes a technique called Optimizer fusion (register the optimizer with model).
    The semantics of this is that during the backwards pass, the registered optimizer
    will be called.

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B x F x D] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        optimizer (Type[torch.optim.Optimizer]): fusion optimizer type
        optimizer_kwargs: Dict[str, Any]: fusion optimizer kwargs
        device (Optional[torch.device]): compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=8, num_embeddings=10, feature_names=["f2"]
        )

        ebc = FusedEmbeddingBagCollection(tables=[table_0, table_1],
        optimizer=torch.optim.SGD, optimizer_kwargs={"lr": .01})

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[ 0.2093,  0.1395,  0.1571,  0.3583,  0.0421,  0.0037, -0.0692,  0.0663,
          0.2166, -0.3150, -0.2771, -0.0301],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0165, -0.1225,  0.2483,  0.0624,
         -0.1168, -0.0509, -0.1309,  0.3059],
        [ 0.0811, -0.1779, -0.1443,  0.1097, -0.4410, -0.4036,  0.4458, -0.2735,
         -0.3080, -0.2102, -0.0564,  0.5583]], grad_fn=<CatBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        [0, 4, 12]
    """

    def __init__(
        self,
        schema_or_tables: Union[Schema, List[EmbeddingBagConfig]],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        location: Optional[EmbeddingLocation] = None,
        pre=None,
        post=None,
        **table_kwargs
    ) -> None:
        super().__init__(
            self.parse_tables(schema_or_tables, config_cls=EmbeddingBagConfig, **table_kwargs),
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            is_weighted=is_weighted,
            device=device,
            location=location,
        )
        self.register_hooks(pre=pre, post=post)


class FusedEmbeddingCollection(TorchrecFusedEmbeddingCollection, TorchrecEmbeddingMixin):
    """
    EmbeddingCollection represents a unsharded collection of non-pooled embeddings. The semantics
    of this module is that during the backwards pass, the registered optimizer will be called.

    It processes sparse data in the form of `KeyedJaggedTensor` of the form [F X B X L]
    where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (variable)

    and outputs `Dict[feature (key), JaggedTensor]`.
    Each `JaggedTensor` contains values of the form (B * L) X D
    where:

    * B: batch size
    * L: length of sparse features (jagged)
    * D: each feature's (key's) embedding dimension and lengths are of the form L

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup dict.

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([[-0.2050,  0.5478,  0.6054],
        [ 0.7352,  0.3210, -3.0399],
        [ 0.1279, -0.1756, -0.4130],
        [ 0.7519, -0.4341, -0.0499],
        [ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>)
    """

    def __init__(
        self,
        schema_or_tables: Union[Schema, List[EmbeddingConfig]],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
        location: Optional[EmbeddingLocation] = None,
        pre=None,
        post=None,
        **table_kwargs
    ) -> None:
        super().__init__(
            self.parse_tables(schema_or_tables, **table_kwargs),
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            need_indices=need_indices,
            device=device,
            location=location,
        )
        self.register_hooks(pre=pre, post=post)


def _forward_kwargs_to_config(col, config_cls, kwargs):
    arg_spec = inspect.getfullargspec(config_cls)
    supported_kwargs = arg_spec.kwonlyargs
    if arg_spec.defaults:
        supported_kwargs += arg_spec.args[-len(arg_spec.defaults) :]

    table_kwargs = {}
    for key, val in kwargs.items():
        if key in supported_kwargs:
            if isinstance(val, dict):
                if col.name in val:
                    table_kwargs[key] = val[col.name]
                elif col.int_domain.name in val:
                    table_kwargs[key] = val[col.int_domain.name]
            else:
                table_kwargs[key] = val

    return table_kwargs


def create_embedding_configs(
    schema: Schema,
    dim: Optional[Union[Dict[str, int], int]] = None,
    infer_dim_fn: Callable[[ColumnSchema], int] = infer_embedding_dim,
    config_cls: Type[Config] = EmbeddingConfig,
    **kwargs
) -> List[Config]:
    configs = {}

    for col in schema:
        table_name = col.int_domain.name or col.name
        if table_name in configs:
            configs[table_name]["feature_names"] += [col.name]
        else:
            config_kwargs = _forward_kwargs_to_config(col, config_cls, kwargs)
            configs[table_name] = {
                "num_embeddings": col.int_domain.max + 1,
                "embedding_dim": _get_dim(col, dim, infer_dim_fn),
                "name": table_name,
                "feature_names": [col.name],
                **config_kwargs,
            }

    return [config_cls(**config) for config in configs.values()]
