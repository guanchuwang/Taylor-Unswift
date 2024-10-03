from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize(["llama.py",
                               "mistral.py",
                               "mistral.py",
                               "phi.py",
                               "utils.py",
                               ]))