#!/usr/bin/env python
"""
Execute this file in the git-repository with 'python setup.py install' to install pysages.
"""
import sys
from setuptools import setup, find_packages

def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("pysages")

def main(argv):
    """
    Main function organizing and orchastrating the install.
    """

    setup(name='pysages',
          version=version,
          description='PySAGES (Python Suite for Advanced General Ensemble Simulations) is an Python implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.',
          cmdclass=cmdclass,
          install_requires=["jax", "Cython", "plum-dispatch", "fastcore", "numba"],
          author='Pablo Zubieta et. al',
          author_email='pzubieta@uchicago.edu',
          python_requires='>=3.6, <4',
          classifiers=['Development Status :: 3 - Alpha',
                       'Intended Audience :: Developers',
                       'License :: MIT/GPL-3',
                       'Programming Language :: Python :: 3 :: Only',
                       ],
          packages=find_packages(),
          )

    # Test if jax is installed correctly
    try:
        import jax
    except ImportError:
        print("Jax requires `jaxlib` to be installed. You should install jaxlib according to your CUDA version. See https://github.com/google/jax#installation for details.")


if __name__ == "__main__":
    main(sys.argv[1:])
