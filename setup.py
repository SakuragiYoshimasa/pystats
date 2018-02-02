from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import pandas
import numpy

ext_modules = [
    Extension('pystats.anova',
              sources=['pystats/anova.pyx'],
              include_dirs=[numpy.get_include()])
]

setup(
    name='pystats',
    version='0.0.1',
    description='Python package for data analysis.',
    long_description='README.md',
    author='Yoshimasa Sakuragi',
    author_email='ysakuragi16@gmail.com',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib'],
    url='https://github.com/SakuragiYoshimasa/pyplfv',
    license=license,
    ext_modules=ext_modules,
    test_suite='tests',
    cmdclass={'build_ext': build_ext}
)
