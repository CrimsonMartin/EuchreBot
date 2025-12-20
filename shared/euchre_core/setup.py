from setuptools import setup, find_packages

setup(
    name="euchre_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
    ],
    author="EuchreBot Team",
    description="Core Euchre game engine and logic",
    python_requires=">=3.10",
)
