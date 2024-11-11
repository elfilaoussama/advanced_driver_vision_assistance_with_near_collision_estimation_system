from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "processing_cy",
        ["processing.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"]
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[np.get_include()]
)