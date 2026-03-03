from setuptools import find_packages, setup

setup(
    name="grok-qiangjian",
    packages=find_packages(),
    version="0.0.2-qiangjian",
    description=(
        "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. "
        "强兼 (forceful compatibility) bridge: now includes Grok-1 MoE architecture."
    ),
    url="https://github.com/openai/grok",
    install_requires=[
        "pytorch_lightning",
        "blobfile",
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
    ],
)
