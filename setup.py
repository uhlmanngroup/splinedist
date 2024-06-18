#!/usr/bin/env python

"""The setup script."""
from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = ["stardist==0.8.3", "tensorflow==2.10.0", "opencv-python-headless","csbdeep==0.6.3","numpy==1.21.2","numba==0.51.2"]

setup(
    author="uhlmanngroup",
    author_email="uhlmann@ebi.ac.uk",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="splinedist",
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="splinedist",
    name="splinedist",
    packages=find_packages(),
    url="https://github.com/uhlmanngroup/splinedist",
    version="0.1.2",
    zip_safe=False,
)
