"""
Setup script for Treadmill framework.
"""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip().split('#')[0].strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pytorch-treadmill",
    version="0.6.2",
    author="Mayukh Sarkar",
    author_email="mayukh2012@hotmail.com",
    description="A Clean and Modular PyTorch Training Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MayukhSobo/treadmill",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "build>=0.8.0",
            "twine>=4.0.0",
        ],
        "examples": [
            "torchvision>=0.13.0",
            "scikit-learn>=1.0.0",
        ],
        "full": [
            "torchinfo>=1.7.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
            "onnx>=1.12.0",
            "safetensors>=0.3.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ]
    },
    keywords="pytorch, deep learning, training, machine learning, neural networks",
    project_urls={
        "Bug Reports": "https://github.com/MayukhSobo/treadmill/issues",
        "Source": "https://github.com/MayukhSobo/treadmill",
        "Documentation": "https://mayukhsobo.github.io/treadmill/",
    },
) 