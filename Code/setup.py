from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("kickedrotor.bipartite_entanglement",
            sources=["kickedrotor/bipartite_entanglement.pyx"],
            libraries=["m"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"])
]

compiler_directives = {
    "language_level": 3
}

config = {
    "description": "Calculates bipartite entanglement in quasiperiodic kicked rotor",
    "author": "Aditya Chincholi",
    "author_email": "aditya.chincholi@students.iiserpune.ac.in",
    "version": 0.1,
    "install_requires": ["numpy", "scipy", "matplotlib", "nose"],
    "packages": ["kickedrotor"],
    "scripts": [],
    "name": 'Quasiperiodic Kicked Rotor Entanglement',
    "ext_modules": cythonize(ext_modules, annotate=True, gdb_debug=True,
                                compiler_directives=compiler_directives),
    "zip_safe": False
}

setup(**config)
