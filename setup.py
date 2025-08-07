from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function will return the list of requirements'''
    requirements=[]
    with open(file_path) as file_obj:
        reqiurements = file_obj.readlines()
        [req.replace("\n", ' ') for req in requirements]

        if HYPEN_E_DOT in reqiurements: 
            reqiurements.remove(HYPEN_E_DOT)

    return requirements



setup(
    name='Chun_prediction',
    version='0.0.2',
    author='Tanvir',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)


