# Module: PySDMs
# Author: Daniel Ryan Furman <dryanfurman@gmail.com>
# License: MIT
# Last modified : 4/7/21

# https://github.com/daniel-furman/PySDMs

import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="PySDMs",
    version="0.1",
    author="Daniel Ryan Furman",
    author_email="dryanfurman@gmail.com",
    description=("An object-oriented class for ML geo-classification."),
    long_description="See documentation at https://github.com/daniel-furman/PySDMs",
    license="MIT",
    keywords="species-distribution-modeling machine-learning biodiversity",
    url="https://github.com/daniel-furman/PySDMs",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "pandas", "sklearn", "matplotlib", "pycaret",
        "pyimpute", "geopandas", "rasterio"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
        ],
)
