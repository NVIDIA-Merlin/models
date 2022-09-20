#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
from typing import Dict, Union

import fiddle as fdl


def extract_hparams_from_config(config: fdl.Config) -> Dict[str, Union[str, Dict]]:
    """Extracts a dict with the configurations from a Fiddle config

    Parameters
    ----------
    config : fdl.Config
        A Fiddle config object

    Returns
    -------
    Dict[str, Union[str, Dict]]
        Dict hierarchy with the config defined args and also the default args
        from the configured callable (function or class constructor)
    """
    output_args = {}
    config_args = config.__arguments__
    for k, v in config_args.items():
        if isinstance(v, fdl.Config):
            output_args[k] = extract_hparams_from_config(v)

        else:
            output_args[k] = v

    callable_params = inspect.signature(config.__fn_or_cls__).parameters
    for k, v in callable_params.items():
        if k not in output_args and callable_params[k].default is not callable_params[k].empty:
            output_args[k] = callable_params[k].default
    return output_args
