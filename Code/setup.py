from setuptools import setup
from Cython.Build import cythonize

config = {
    "description": "Calculates bipartite entanglement in quasiperiodic kicked rotor",
    "author": "Aditya Chincholi",
    "author_email": "aditya.chincholi@students.iiserpune.ac.in",
    "version": 0.1,
    "install_requires": ["numpy", "scipy", "matplotlib", "nose"],
    "packages": ["kickedrotor"],
    "scripts": [],
    "name": 'Quasiperiodic Kicked Rotor Entanglement',
    "ext_modules": cythonize("kickedrotor/bipartite_entanglement.pyx"),
    "zip_safe": False,
}

setup(**config)
