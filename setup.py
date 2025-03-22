import os
import sys
import shutil
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.install import install

# Change name to "gaiacmdfit" when you want to
#  load to PYPI
#pypiname = 'gaiacmdfit'

setup(name="gaiacmdfit",
      version='1.0.0',
      description='Fit Gaia CMDs with isochrone models',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/gaiacmdfit',
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils'],
      zip_safe = False,
      include_package_data=True,
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"}
)
