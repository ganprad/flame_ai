""" Setup file for project."""

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    license="MIT",
    description="A package for predicting conditions from src data",
    author="Pradeep Ganesan",
    packages=find_packages(where="src"),
)
