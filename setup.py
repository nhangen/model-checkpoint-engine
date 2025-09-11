#!/usr/bin/env python3
"""Setup script for Model Checkpoint Engine"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="model-checkpoint-engine",
    version="0.1.0",
    author="Contributors",
    description="Comprehensive checkpoint management and experiment tracking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nhangen/model-checkpoint-engine",
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
        "Topic :: Database :: Database Engines/Servers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "sqlalchemy>=1.4.0",
        "sqlite3",  # Built into Python
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "jinja2>=3.0.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "postgresql": ["psycopg2-binary>=2.9.0"],
        "mysql": ["PyMySQL>=1.0.0"],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "web": [
            "flask>=2.0.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "checkpoint-engine=model_checkpoint.cli:main",
        ],
    },
)