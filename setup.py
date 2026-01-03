"""Setup script for slm-trainer package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="slm-trainer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for training and using small language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/slm-trainer",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15.0",
        "keras>=3.10.0",
        "pandas>=2.2.2",
        "numpy>=2.0.2",
        "jax>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pylint>=2.6.0",
        ],
    },
)

