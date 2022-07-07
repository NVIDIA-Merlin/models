import collections
import contextlib
import threading

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import composite_tensor, type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import tf_inspect

from merlin.schema import Schema

_registry = {}  # Mapping from (python pkg, class name) -> class.


class AutoCompositeTensorTypeSpec(type_spec.BatchableTypeSpec):
    """A tf.TypeSpec for `AutoCompositeTensor` objects."""

    __slots__ = (
        "_param_specs",
        "_non_tensor_params",
        "_omit_kwargs",
        "_prefer_static_value",
        "_callable_params",
        "_serializable",
        "_comparable",
    )

    def __init__(
        self,
        param_specs,
        non_tensor_params,
        omit_kwargs,
        prefer_static_value,
        non_identifying_kwargs,
        callable_params=None,
    ):
        """Initializes a new `_AutoCompositeTensorTypeSpec`.
        Args:
          param_specs: Python `dict` of `tf.TypeSpec` instances that describe
            kwargs to the `AutoCompositeTensor`'s constructor that are `Tensor`-like
            or `CompositeTensor` subclasses.
          non_tensor_params: Python `dict` containing non-`Tensor` and non-
            `CompositeTensor` kwargs to the `AutoCompositeTensor`'s constructor.
          omit_kwargs: Python `tuple` of strings corresponding to the names of
            kwargs to the `AutoCompositeTensor`'s constructor that should be omitted
            from the `_AutoCompositeTensorTypeSpec`'s serialization, equality/
            compatibility checks, and rebuilding of the `AutoCompositeTensor` from
            `Tensor` components.
          prefer_static_value: Python `tuple` of strings corresponding to the names
            of `Tensor`-like kwargs to the `AutoCompositeTensor`s constructor that
            may be stored as static values, if known. These are typically shapes or
            axis values.
          non_identifying_kwargs: Python `tuple` of strings corresponding to the
            names of kwargs to the `AutoCompositeTensor`s constructor whose values
            are not relevant to the unique identification of the
            `_AutoCompositeTensorTypeSpec` instance. Equality/comparison checks and
            `__hash__` do not depend on these kwargs.
          callable_params: Python `dict` of callable kwargs to the
            `AutoCompositeTensor`'s constructor that do not subclass
            `CompositeTensor`, or `None`. If `callable_params` is a non-empty
            `dict`, then serialization of the `_AutoCompositeTensorTypeSpec` is not
             supported. Defaults to `None`, which is converted to an empty `dict`.
        """
        self._param_specs = param_specs
        self._non_tensor_params = non_tensor_params
        self._omit_kwargs = omit_kwargs
        self._prefer_static_value = prefer_static_value
        self._non_identifying_kwargs = non_identifying_kwargs
        self._callable_params = {} if callable_params is None else callable_params

        self._serializable = (
            self._param_specs,
            self._non_tensor_params,
            self._omit_kwargs,
            self._prefer_static_value,
            self._non_identifying_kwargs,
        )

        def remove_kwargs(d):
            return {k: v for k, v in d.items() if k not in self._non_identifying_kwargs}

        self._comparable = (
            remove_kwargs(self._param_specs),
            remove_kwargs(self._non_tensor_params),
            self._omit_kwargs,
            self._prefer_static_value,
            self._non_identifying_kwargs,
            tf.nest.map_structure(id, remove_kwargs(self._callable_params)),
        )

    @classmethod
    def from_instance(cls, instance, omit_kwargs=(), non_identifying_kwargs=()):
        cls_value_type = cls.value_type.fget(None)
        if type(instance) is not cls_value_type:  # pylint: disable=unidiomatic-typecheck
            raise ValueError(
                f"`{type(instance).__name__}` has inherited the "
                f"`_type_spec` of `{cls_value_type.__name__}`. It "
                f"should define its own, either directly, or by "
                f"applying `auto_composite_tensor` to "
                f"`{type(instance).__name__}.`"
            )
        prefer_static_value = tuple(getattr(instance, "_composite_tensor_shape_params", ()))
        kwargs = _extract_init_kwargs(
            instance, omit_kwargs=omit_kwargs, prefer_static_value=prefer_static_value
        )

        non_tensor_params = {}
        param_specs = {}
        callable_params = {}
        for k, v in list(kwargs.items()):
            # If v contains no Tensors, this will just be v
            type_spec_or_v = _extract_type_spec_recursively(v)
            if type_spec_or_v and type_spec_or_v is not v:
                param_specs[k] = type_spec_or_v
            elif callable(v):
                callable_params[k] = v
            else:
                non_tensor_params[k] = v

        # Construct the spec.
        return cls(
            param_specs=param_specs,
            non_tensor_params=non_tensor_params,
            omit_kwargs=omit_kwargs,
            prefer_static_value=prefer_static_value,
            non_identifying_kwargs=non_identifying_kwargs,
            callable_params=callable_params,
        )

    def _to_components(self, obj):
        return _extract_init_kwargs(obj, limit_to=list(self._param_specs))

    def _from_components(self, components):
        kwargs = dict(self._non_tensor_params, **self._callable_params, **components)
        with _deferred_assertion_context():
            return self.value_type(**kwargs)

    @property
    def _component_specs(self):
        return self._param_specs

    def _serialize(self):
        if self._callable_params:
            raise ValueError(
                f"Cannot serialize object with callable parameters that are not "
                f"`CompositeTensor`s: {self._callable_params.keys()}."
            )
        return self._serializable

    @classmethod
    def _deserialize(cls, encoded):
        return cls(*encoded)

    def most_specific_compatible_type(self, other):
        """Returns the most specific TypeSpec compatible with `self` and `other`.
        Args:
          other: A `TypeSpec`.
        Raises:
          ValueError: If there is no TypeSpec that is compatible with both `self`
            and `other`.
          ValueError: If the `_callable_params` attributes of `self` and `other` are
            not equal.
        """
        if type(self) is not type(other):
            raise ValueError(f"No TypeSpec is compatible with both {self} and {other}.")
        # pylint: disable=protected-access
        if self._callable_params != other._callable_params:
            raise ValueError(
                f"Callable parameters must be identical. Saw "
                f"{self._callable_params} and {other._callable_params}."
            )
        merged = self._TypeSpec__most_specific_compatible_type_serialization(
            self._comparable[:-1], other._comparable[:-1]
        )
        # pylint: enable=protected-access
        return type(self)(*merged[1:], self._callable_params)

    def is_compatible_with(self, spec_or_value):
        """Returns true if `spec_or_value` is compatible with this TypeSpec."""
        if not isinstance(spec_or_value, tf.TypeSpec):
            spec_or_value = type_spec.type_spec_from_value(spec_or_value)
        if type(self) is not type(spec_or_value):
            return False
        return self._TypeSpec__is_compatible(
            self._comparable, spec_or_value._comparable
        )  # pylint: disable=protected-access

    def _copy(self, **overrides):
        kwargs = {
            "param_specs": self._param_specs,
            "non_tensor_params": self._non_tensor_params,
            "omit_kwargs": self._omit_kwargs,
            "prefer_static_value": self._prefer_static_value,
            "non_identifying_kwargs": self._non_identifying_kwargs,
            "callable_params": self._callable_params,
        }
        kwargs.update(overrides)
        return type(self)(**kwargs)

    def _with_tensor_ranks_only(self):
        """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.
        Returns:
          A `TypeSpec` that is compatible with `self`, where any `TensorShape`
          information has been relaxed to include only tensor rank (and not
          the dimension sizes for individual axes).
        """

        def relax(value):
            if isinstance(value, tf.TypeSpec):
                return value._with_tensor_ranks_only()  # pylint: disable=protected-access
            elif isinstance(value, tf.TensorShape) and value.rank is not None:
                return tf.TensorShape([None] * value.rank)
            else:
                return value

        return self._copy(param_specs=tf.nest.map_structure(relax, self._param_specs))

    def __get_cmp_key(self):
        comparable = []
        for comp in self._comparable:
            if "schema" in comp and isinstance(comp["schema"], Schema):
                # comp["schema"] = str(TensorflowMetadata.from_merlin_schema(
                #     comp["schema"]
                # ).proto_schema.__bytes__())
                del comp["schema"]

            comparable.append(self._TypeSpec__make_cmp_key(comp))

        key = (type(self), tuple(comparable))

        return key

    def __repr__(self):
        return "%s%r" % (type(self).__name__, self._serializable + (self._callable_params,))

    def __reduce__(self):
        if self._callable_params:
            raise ValueError(
                f"Cannot serialize object with callable parameters that are not "
                f"`CompositeTensor`s: {self._callable_params.keys()}."
            )
        return super(AutoCompositeTensorTypeSpec, self).__reduce__()

    def __eq__(self, other):
        self_key = self.__get_cmp_key()
        other_key = other.__get_cmp_key()
        # self_key = self._serialize()
        # other_key = other._serialize()

        return type(other) is type(self) and self_key == other_key

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__get_cmp_key())

    def _batch(self, batch_size):
        """Returns a TypeSpec representing a batch of objects with this TypeSpec."""
        # This method recursively adds a batch dimension to all parameter Tensors.
        # Note that this may result in parameter shapes that do not broadcast. You
        # may wish to first call
        # `dist = dist._broadcast_parameters_with_batch_shape(tf.ones_like(
        # `dist.batch_shape_tensor()))` to ensure that the parameters of a
        # Distribution or analogous object will continue to broadcast after
        # batching.
        return self._copy(
            param_specs=tf.nest.map_structure(
                lambda spec: spec._batch(batch_size),  # pylint: disable=protected-access
                self._param_specs,
            )
        )

    def _unbatch(self):
        """Returns a TypeSpec representing a single element of this TypeSpec."""
        return self._copy(
            param_specs=tf.nest.map_structure(
                lambda spec: spec._unbatch(), self._param_specs  # pylint: disable=protected-access
            )
        )

    @classmethod
    def name(cls):
        return f"{cls.__module__}.{cls.__name__}"


def _extract_init_kwargs(obj, omit_kwargs=(), limit_to=None, prefer_static_value=()):
    """Extract constructor kwargs to reconstruct `obj`."""
    # If `obj` inherits its constructor from `AutoCompositeTensor` (which inherits
    # its constructor from `object`) return an empty dictionary to avoid
    # triggering the error below due to *args and **kwargs in the constructor.
    if type(obj).__init__ is AutoCompositeTensor.__init__:
        return {}

    sig = _cached_signature(type(obj).__init__)
    if any(
        v.kind in (tf_inspect.Parameter.VAR_KEYWORD, tf_inspect.Parameter.VAR_POSITIONAL)
        for v in sig.parameters.values()
    ):
        raise ValueError("*args and **kwargs are not supported. Found `{}`".format(sig))

    keys = [p for p in sig.parameters if p != "self" and p not in omit_kwargs]
    if limit_to is not None:
        keys = [k for k in keys if k in limit_to]

    kwargs = {}
    not_found = object()
    for k in keys:
        src1 = getattr(obj, k, not_found)
        if src1 is not not_found:
            kwargs[k] = src1
        else:
            src2 = getattr(obj, "_" + k, not_found)
            if src2 is not not_found:
                kwargs[k] = src2
            else:
                src3 = getattr(obj, "parameters", {}).get(k, not_found)
                if src3 is not not_found:
                    kwargs[k] = src3
                else:
                    raise ValueError(
                        f"Could not determine an appropriate value for field `{k}` in"
                        f" object `{obj}`. Looked for \n"
                        f" 1. an attr called `{k}`,\n"
                        f" 2. an attr called `_{k}`,\n"
                        f' 3. an entry in `obj.parameters` with key "{k}".'
                    )
        if k in prefer_static_value and kwargs[k] is not None:
            if tf.is_tensor(kwargs[k]):
                static_val = tf.get_static_value(kwargs[k])
                if static_val is not None:
                    kwargs[k] = static_val
        if isinstance(kwargs[k], (np.ndarray, np.generic)):
            # Generally, these are shapes or int, but may be other parameters such as
            # `power` for `tfb.PowerTransform`.
            kwargs[k] = kwargs[k].tolist()

    return kwargs


def _extract_type_spec_recursively(value):
    """Return (collection of) TypeSpec(s) for `value` if it includes `Tensor`s.
    If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If
    `value` is a collection containing `Tensor` values, recursively supplant them
    with their respective `TypeSpec`s in a collection of parallel structure.
    If `value` is nont of the above, return it unchanged.
    Args:
      value: a Python `object` to (possibly) turn into a (collection of)
      `tf.TypeSpec`(s).
    Returns:
      spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`
      or `value`, if no `Tensor`s are found.
    """
    if isinstance(value, composite_tensor.CompositeTensor):
        return value._type_spec  # pylint: disable=protected-access
    if isinstance(value, tf.Variable):
        return resource_variable_ops.VariableSpec(
            value.shape, dtype=value.dtype, trainable=value.trainable
        )
    if tf.is_tensor(value):
        return tf.TensorSpec(value.shape, value.dtype)
    if isinstance(value, collections.Mapping):
        return tf.nest.map_structure(_extract_type_spec_recursively, value)
    if isinstance(value, (list, tuple)):
        specs = [_extract_type_spec_recursively(v) for v in value]
        was_tensor = list([a is not b for a, b in zip(value, specs)])
        has_tensors = any(was_tensor)
        has_only_tensors = all(was_tensor)
        if has_tensors:
            if has_tensors != has_only_tensors:
                raise NotImplementedError(
                    "Found `{}` with both Tensor and non-Tensor parts: {}".format(
                        type(value), value
                    )
                )
            return type(value)(specs)
    return


# Cache maps __init__ method to signature
_sig_cache = {}


def _cached_signature(f):
    if f not in _sig_cache:
        _sig_cache[f] = tf_inspect.signature(f)
    return _sig_cache[f]


_DEFERRED_ASSERTION_CONTEXT = threading.local()
_DEFERRED_ASSERTION_CONTEXT.is_deferred = False


def is_deferred_assertion_context():
    return getattr(_DEFERRED_ASSERTION_CONTEXT, "is_deferred", False)


@contextlib.contextmanager
def _deferred_assertion_context(is_deferred=True):
    was_deferred = getattr(_DEFERRED_ASSERTION_CONTEXT, "is_deferred", False)
    _DEFERRED_ASSERTION_CONTEXT.is_deferred = is_deferred
    try:
        yield
    finally:
        _DEFERRED_ASSERTION_CONTEXT.is_deferred = was_deferred


class AutoCompositeTensor(composite_tensor.CompositeTensor):
    """Recommended base class for `@auto_composite_tensor`-ified classes.
    See details in `tfp.experimental.auto_composite_tensor` description.
    """

    @property
    def _type_spec(self):
        # This property will be overwritten by the `@auto_composite_tensor`
        # decorator. However, we need it so that a valid subclass of the `ABCMeta`
        # class `CompositeTensor` can be constructed and passed to the
        # `@auto_composite_tensor` decorator
        pass
