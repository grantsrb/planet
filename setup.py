from setuptools import setup, find_packages

setup(name='planet',
      version="0.1.0",
      description='Implementation of PlaNet',
      author='Satchel Grant',
      author_email='grantsrb@gmail.com',
      url='https://github.com/grantsrb/planet.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          The planet package contains methods for implementing the algorithms in the
          paper: Learning Latent Dynamics from Pixels for Planning.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
      )
