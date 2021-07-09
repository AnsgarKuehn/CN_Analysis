from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    

setup(name = 'cn_tools',
    version = '0.1',
    author = 'Ansgar Kühn',
    author_email = 'ansgar.kuehn@posteo.de',
    description = 'This package contains helpful functions for the processing and evaluation of data the Minkowski tensors of monidisperse spheres obtained from Karambola.',
    install_requires=requirements,
    packages=find_packages(),
    )
