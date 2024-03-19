from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning==1.5.10",
        "blobfile",
        "numpy==1.23.0",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
    ],
)
