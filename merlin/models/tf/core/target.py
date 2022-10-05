from typing import Dict, Optional

import tensorflow as tf
from keras.engine import base_layer_utils


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


def clear_targets(layer):
    """For use in the model"""
    
    # Maintains info about the `Layer.call` stack.
    call_context = base_layer_utils.call_context()
    
    if not call_context.in_call:
        if not getattr(
            layer, "_self_tracked_trackables", None
        ):  # Fast path for single Layer.
            layer._thread_local._eager_targets = {}
        else:
            for layer in layer._flatten_layers():
                layer._thread_local._eager_targets = {}   


class TargetLayer(tf.keras.layers.Layer):
    def __call__(self, *args, return_targets=False, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        
        if return_targets:
            targets = get_targets(self, kwargs.get("targets", {}))
            
            return outputs, targets
        
        return outputs
	
    def compute_target(self, name: Optional[str], target):
        if not getattr(self, "_targets", None):
            self._targets = {}
        self._targets[name] = target

    @property
    def targets(self):
        return layer_targets(self)
