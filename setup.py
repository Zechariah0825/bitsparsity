from setuptools import setup, Extension
import numpy

setup(
    name='bitmod',
    version='1.0',
    ext_modules=[
        Extension('bitmod', ['bitmod.cpp'],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['-std=c++11'])  
    ],
)