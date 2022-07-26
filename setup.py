import os
import setuptools

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='AI-lib',
    version='0.1.0',
    license="MIT",
    long_description=long_description,
    author='Jussi Kärkkäinen',
    url="https://github.com/JussiKarkkainen/AIlib",
    packages=setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires= '>=3.8',
)   

