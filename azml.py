import logging
import os
import sys
from cv_lib.azureml_tools import workspace
from cv_lib.azureml_tools import PyTorchExperiment
from cv_lib.azureml_tools.config import experiment_config
from cv_lib.azureml_tools.experiment import create_environment_from_local
from invoke import run


def _create_wheel():
    logger = logging.getLogger("__name__")
    cmd = f"cd cv_lib && {sys.executable} setup.py bdist_wheel"
    print(f"Running {cmd}")
    result = run(cmd, hide=True, warn=True)
    print(result.ok)
    print(result.stdout.splitlines()[-1])
    if result.ok is False:
        raise UserWarning("Failed to create wheel")

def main():
    print(experiment_config)
    exp = PyTorchExperiment("pytorch",config=experiment_config)
    env = create_environment_from_local(name='amlenv', conda_env_name=None)
    _create_wheel()
    cv_lib_wheel = os.path.abspath("cv_lib/dist/cv_lib-0.0.1-py3-none-any.whl")
    cv_lib = env.add_private_pip_wheel(exp._ws, cv_lib_wheel, exist_ok=True)
    env.python.conda_dependencies.add_pip_package(cv_lib)
    print("Environment")
    print(env.python.conda_dependencies.serialize_to_string())

    run = exp.submit(
            os.path.abspath("."),
            "pytorch_synthetic_benchmark.py",
            {
                "--node_rank": "$AZ_BATCHAI_TASK_INDEX",
                "--dist-url": "$AZ_BATCHAI_PYTORCH_INIT_METHOD",
                "--world-size": 4 * 2,
                "--ngpus_per_node": 4,
                "--model": resnet50,
                "--batch-size": 64,
                "--dataset": "synthetic",
                # "DATASET.ROOT":"{datastore}/penobscot",
                # "MODEL.PRETRAINED":"{datastore}/hrnet_pretrained/image_classification/hrnetv2_w48_imagenet_pretrained.pth",
                # "--cfg": "configs/hrnet.yaml",
                
            },
            node_count=2,
            distributed="nccl",
            environment=env
        )




