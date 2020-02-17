import os
import re

from setuptools import setup, find_packages


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


def find_version(file):
    content = read_file(file)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


required = resolve_requirements(
    os.path.join(os.path.dirname(__file__), 'requirements.txt'))
required_experiment = resolve_requirements(
    os.path.join(os.path.dirname(__file__), 'requirements_experiment.txt'))
readme = read_file(
    os.path.join(os.path.dirname(__file__), "README.md"))
version = find_version(
    os.path.join(os.path.dirname(__file__), "neuralprocess", "__init__.py"))


setup(name='neuralprocess',
      version=version,
      description='Neural Processes in PyTorch',
      long_description=readme,
      long_description_content_type="text/markdown",
      author='Anonymous Authors',
      author_email='anon.email@domain.com',
      license="MIT",
      packages=find_packages(),
      install_requires=required,
      zip_safe=True,
      include_package_data=True,
      extras_require={
          "experiment": required_experiment
      }
      )
