from setuptools import find_packages, setup
from typing import List

REQUIREMENTS_FILENAME = 'requirements.txt'


def get_requirements_list()->List[str]:
    """
    This function is going to return list of requirements present in requirements.txt file

    returns a list of all library names needed to be installed to run the app.
    """
    with open(REQUIREMENTS_FILENAME, 'r') as requirements_file:
        return requirements_file.readlines().remove('-e .')

setup(
    name = 'concrete-strength-estimator',
    version='0.0.2',
    author='Abhinesh Kourav',
    description='Estimate concrete compressive strength without waiting for 28 days like in conventional UCS test.',
    packages=find_packages(),
    install_requires = get_requirements_list()
)