from setuptools import setup
from setuptools import find_packages


__version__ = "0.0.0"


setup(
    name='mlprague22',
    package_dir={'': 'src'},
    include_package_data=False,
    packages=find_packages('src'),
    version=__version__,
    description='ML Prague 2022 recommender systems workshop',
    author='Seznam.cz, a.s.',
    author_email='tomas.novacik@firma.seznam.cz',
    url='git@github.com/seznam/MLPrague-2022.git',
    license='Proprietary',
    install_requires=[]

)
