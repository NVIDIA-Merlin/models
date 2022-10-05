from typing import Dict, Optional
from time import time

import tensorflow as tf


def get_targets(layer, targets=None):
    if not targets:
        targets = {}
    
    _targets = getattr(layer, "targets", None)
    if _targets:
        for key, val in _targets.items():
            targets[key] = val

    return targets


def layer_targets(layer) -> Dict[str, tf.Tensor]:
    collected_targets = {}

    for child in layer._flatten_layers():
        layer_targets = getattr(child, "_targets", None)
        layer_targets_time = getattr(child, "_targets_time", None)
        if layer_targets:
            for key, val in layer_targets.items():
                if key in collected_targets:
                    if layer_targets_time[key] > collected_targets[key][-1]:
                        collected_targets[key] = (val, time)
                else:
                    collected_targets[key] = (val, time)

    targets = {key: val[0] for key, val in collected_targets.items()}

    return targets


class TargetLayer(tf.keras.layers.Layer):
    def __call__(self, *args, return_targets=False, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        
        if return_targets:
            targets = get_targets(self, kwargs.get("targets", {}))
            
            return outputs, targets
        
        return outputs
	
    def compute_target(self, name: Optional[str], target):
        if not getattr(self, "_targets", None):
            self._targets, self._targets_time = {}, {}
        self._targets[name] = target
        self._targets_time[name] = tf.constant(time())

    @property
    def targets(self):
        return layer_targets(self)
