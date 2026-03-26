from setuptools import setup, find_packages

setup(
    name="ddpa-img-preprocessing",
    version="0.1.0",
    description="Preprocessing pipeline for medieval charter images in FSDB",
    author="anguelos",
    license="AGPL-3.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "Pillow>=9.0",
        "numpy>=1.22",
        "tqdm>=4.64",
        "fargv>=0.1",
        "torch>=2.0",
        "torchvision>=0.15",
        "torch_mentor",
        "rarfile>=4.0",
        "py7zr>=0.20",
        "gdown>=4.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "sphinx>=6.0",
            "myst-parser>=1.0",
            "sphinx-rtd-theme>=1.2",
            "ruff>=0.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "ddp_binarize_offline=ddp_binarize.binarize:main_binarize_offline",
            "ddp_res_offline=ddp_resolution.resolution:main_resolution_offline",
            "ddp_recto_offline=ddp_recto.recto_verso:main_recto_verso_offline",
            "ddp_cv_preprocess_offline=ddp_cv_preprocess.offline:main_cv_preprocess_offline",
            "ddp_res_evaluate=ddp_resolution.res_ds:main_res_evaluate",
        ]
    },
)
