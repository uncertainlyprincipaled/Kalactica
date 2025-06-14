"""Setup script for KaLactica."""

from setuptools import setup, find_packages

setup(
    name="kalactica",
    version="0.1.0",
    description="A Memory- & Topology-Enhanced Successor to Galactica",
    author="KaLactica Team",
    author_email="team@kalactica.org",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "gudhi>=3.7.0",  # Optional, for topology features
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "kalactica=kalactica.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 