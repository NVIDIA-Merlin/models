from typing import Dict, List, Optional
import six
import sys

import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.engine import base_layer_utils

from merlin.models.tf.core.base import block_registry
from merlin.models.tf.utils import tf_utils


class TargetMixin:
    def compute_target(self, name: Optional[str], target):
        if not getattr(self, "_targets", None):
            self._targets = {}
        self._targets[name] = target
        
    def clear_targets(self, include_self=True):
        if not getattr(
            self, "_self_tracked_trackables", None
        ):  # Fast path for single Layer.
            self._thread_local._eager_targets = {}
        else:
            for layer in self._flatten_layers(include_self=include_self):
                layer._thread_local._eager_targets = {}
                
    def call_sequentially(self, to_call: List[Layer], inputs, **kwargs):
        """Call layers sequentially."""

        outputs = call_sequentially(to_call, inputs, **kwargs)
        
        # Update targets, when layer is a target-producer
        layer_targets = get_targets(self)
        for key, val in layer_targets.items():
            self.compute_target(key, val)
            self.clear_targets(include_self=False)

        return outputs
    
    def call_parallel(self, to_call: Dict[str, Layer], inputs, **kwargs):
        outputs = {}

        for name, branch in to_call.items():
            branch_out = tf_utils.call_layer(branch, inputs, **kwargs)
            if isinstance(branch_out, dict):
                outputs.update(branch_out)
            else:
                outputs[name] = branch_out

            # Update targets, when layer is a target-producer
            branch_targets = get_targets(branch)
            for key, val in branch_targets.items():
                if key in get_targets(self):
                    raise ValueError(f"Multiple branches tried to transform: {key}")
                self.compute_target(key, val)
                self.clear_targets(include_self=False)
                
        return outputs
    
    @property
    def targets(self) -> Dict[str, tf.Tensor]:
        targets = {}

        for child in self._flatten_layers():
            layer_targets = getattr(child, "_targets", None)
            if layer_targets:
                for key, val in layer_targets.items():
                    if key in targets:
                        raise ValueError(
                            f"Multiple layers transformed the same target: {key}. ",
                            "Please make sure that the target is bound to the outermost layer.",
                        )
                    else:
                        targets[key] = val

        return targets


class BlockV2(Layer, TargetMixin):
    registry = block_registry
    
    def __init__(
        self, 
        pre: Optional[Layer] = None,
        post: Optional[Layer] = None,
        trainable=True, 
        name=None, 
        dtype=None, 
        dynamic=False, 
        **kwargs
    ):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.pre = pre
        self.post = post
        
    def __call__(self, *args, return_targets=False, **kwargs):
        if self.pre:
            inputs = tf_utils.call_layer(self.pre, inputs, **kwargs)
        
        outputs = super().__call__(*args, **kwargs)
        
        if self.post:
            outputs = tf_utils.call_layer(self.post, outputs, **kwargs)
        
        if return_targets:            
            targets = get_targets(self, kwargs.get("targets", {}))
            clear_targets(self)
            
            return outputs, targets
        
        return outputs
    
    @property
    def to_call_generator(self):
        if self.pre:
            yield self.pre
            
        yield self
        
        if self.post:
            yield self.post
            
    @property
    def to_call(self) -> List[Layer]:
        return list(self.to_call_generator)
    
    def get_config(self):
        return tf_utils.maybe_serialize_keras_objects(
            self, 
            super(BlockV2, self).get_config(), 
            ["pre", "post"]
        )

    @classmethod
    def from_config(cls, config):
        config = tf_utils.maybe_deserialize_keras_objects(
            config,
            {
                "pre": tf.keras.layers.deserialize,
                "post": tf.keras.layers.deserialize,
            },
        )

        return super().from_config(config)


def get_targets(layer, targets=None):
    if not targets:
        targets = {}
    
    _targets = getattr(layer, "targets", None)
    if _targets:
        for key, val in _targets.items():
            targets[key] = val

    return targets


def layer_targets(layer) -> Dict[str, tf.Tensor]:
    targets = {}

    for child in layer._flatten_layers():
        layer_targets = getattr(child, "_targets", None)
        if layer_targets:
            for key, val in layer_targets.items():
                if key in targets:
                    raise ValueError(
                        f"Multiple layers transformed the same target: {key}. ",
                        "Please make sure that the target is bound to the outermost layer.",
                    )
                else:
                    targets[key] = val

    return targets


def clear_targets(layer, include_self=True):
    if not getattr(
        layer, "_self_tracked_trackables", None
    ):  # Fast path for single Layer.
        layer._thread_local._eager_targets = {}
    else:
        for layer in layer._flatten_layers(include_self=include_self):
            layer._thread_local._eager_targets = {}
                
                
def clear_targets_top_batch(layer):
    """For use in the model"""
    
    # Maintains info about the `Layer.call` stack.
    call_context = base_layer_utils.call_context()
    
    if not call_context.in_call:
        clear_targets(layer)
                
                
def call_sequentially(to_call, inputs, **kwargs):
    """Call layers sequentially."""

    outputs = inputs
    for layer in to_call:
        outputs = tf_utils.call_layer(layer, outputs, **kwargs)

    return outputs


def build_sequentially(self, layers, input_shape):
    """Build layers sequentially."""
    last_layer = None
    for layer in layers:
        try:
            layer.build(input_shape)
        except TypeError:
            t, v, tb = sys.exc_info()
            
            from merlin.models.tf.core.tabular import TabularBlock
            
            if isinstance(input_shape, dict) and isinstance(last_layer, TabularBlock):
                v = TypeError(
                    f"Couldn't build {layer}, "
                    f"did you forget to add aggregation to {last_layer}?"
                )
            six.reraise(t, v, tb)
        input_shape = layer.compute_output_shape(input_shape)
        last_layer = layer
    self.built = True


def compute_output_signature_sequentially(layers, input_signature):
    """Compute output signature sequentially."""
    output_signature = input_signature
    for layer in layers:
        output_signature = layer.compute_output_signature(output_signature)

    return output_signature


def compute_output_shape_sequentially(layers, input_shape):
    """Compute output shape sequentially."""
    output_shape = input_shape
    for layer in layers:
        output_shape = layer.compute_output_shape(output_shape)

    return output_shape