"""
Setup file for quantum_noise package.
"""

from setuptools import setup, find_packages

setup(
    name="quantum_noise",
    version="0.1.0",
    description="Single-qubit quantum noise channels from scratch using only NumPy",
    author="Quantum Engineer",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="quantum computing, quantum noise, density matrices, Kraus operators",
    project_urls={
        "Documentation": "https://github.com/shaukat456/quantum_noise",
        "Source": "https://github.com/shaukat456/quantum_noise",
    },
)
