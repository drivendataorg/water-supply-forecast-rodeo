import importlib
import subprocess
import warnings

import pytest


packages = [
    # wsfr-read modules
    "wsfr_read.climate",
    "wsfr_read.streamflow",
    "wsfr_read.teleconnections",
    "wsfr_read.sites",
    # Additional packages
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "tensorflow",
    "torch",
    "xgboost",
    # Requested packages
    "catboost",
    "quantile_forest",
]


def is_gpu_available():
    try:
        return subprocess.check_call(["nvidia-smi"]) == 0

    except FileNotFoundError:
        return False


GPU_AVAILABLE = is_gpu_available()


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    """Test that certain dependencies are importable."""
    importlib.import_module(package_name)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
def test_allocate_torch():
    import torch

    assert torch.cuda.is_available()

    torch.zeros(1).cuda()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
def test_allocate_tf():
    import tensorflow as tf

    assert tf.test.is_built_with_cuda()
    assert (devices := tf.config.list_logical_devices("GPU"))

    for device in devices:
        with tf.device(device.name):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
