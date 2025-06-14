"""Setup script for KaLactica."""

from setuptools import setup, find_packages

setup(
    name="kalactica",
    version="0.1.0",
    description="A topological approach to notebook generation",
    author="Jamie Morgan",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "faiss-cpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "jupyter>=1.0.0",
        "nbformat>=5.7.0",
        "boto3>=1.26.0",
        "opensearch-py>=2.0.0",
        "requests-aws4auth>=1.1.0",
        "gudhi>=3.7.0",
        "tqdm>=4.65.0",
        "peft>=0.4.0",
        "python-dotenv>=1.0.0",
        "awscli>=1.29.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0"
        ]
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