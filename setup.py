import sys
from setuptools import setup, find_packages

setup(
    # Needs to be automatized for each user in a next iteration
    name='pysages',
    author='User',
    author_email='jondoe@uni.edu',
    license='BSD-3-Clause',

    # Search PySAGES module
    packages=find_packages(),

    include_package_data=True,

)


