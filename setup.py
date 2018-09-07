from setuptools import setup
from Cython.Build import cythonize

setup(
    python_requires='>=3.5',
    name='low_level_tools',
    ext_modules=cythonize("low_level_tools.pyx"),
    zip_safe=False,
)
