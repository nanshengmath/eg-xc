from jax import config
import jax
import flax.linen as nn
from .pyscf_wrapper import PyscfSystemWrapper
from .errors import relative_error, is_close, assert_is_close


import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROOT_DATA_DIR = os.path.join(ROOT_DIR, 'data')


def set_jax_testing_config():
    config.update('jax_enable_x64', True)
    config.update('jax_default_matmul_precision', 'float32')
    config.update("jax_platform_name", "cpu")


def call_module_as_function(module: nn.Module, *args, jit=False, **kwargs):
    """
    Calls a Flax module without learnable parameters as a function.
    """
    params = module.init(jax.random.PRNGKey(0), *args, **kwargs)

    def apply(*args, **kwargs):
        return module.apply(params, *args, **kwargs)

    if jit:
        apply = jax.jit(apply)
    out = apply(*args, **kwargs)
    return out