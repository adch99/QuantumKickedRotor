from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("kickedrotor.bipartite_entanglement_windows",
            sources=["kickedrotor/bipartite_entanglement_windows.pyx"],
            libraries=["m"],
            extra_compile_args=["/openmp"],
            extra_link_args=["/openmp", "complex"])
]

config = {
    "description": "Calculates bipartite entanglement in quasiperiodic kicked rotor",
    "author": "Aditya Chincholi",
    "author_email": "aditya.chincholi@students.iiserpune.ac.in",
    "version": 0.1,
    "install_requires": ["numpy", "scipy", "matplotlib", "nose"],
    "packages": ["kickedrotor"],
    "scripts": [],
    "name": 'Quasiperiodic Kicked Rotor Entanglement',
    "ext_modules": cythonize(ext_modules, annotate=True),
    "zip_safe": False,
}

setup(**config)
