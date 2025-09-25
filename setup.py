from setuptools import setup, find_packages

setup(
    name="naive-quantum-circuit",
    version="0.1.0",
    description="A naive quantum circuit simulator",
    author="Gabriel Marques",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "qiskit>=0.45.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)