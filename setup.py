from setuptools import setup, find_packages

setup(
    name="ligafooty",
    version="0.2",
    packages=find_packages(),
    install_requires=[
    "numpy",
    "pandas",
    "matplotlib",
    "mplsoccer",
    "highlight_text",
    "scipy",
    "pillow"
],extras_requires={"dev" : [ "twine"]
},
    author="Ligandro S.Y.",
    description="Python Package to work with football tracking data",
)
