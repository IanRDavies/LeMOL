from setuptools import setup, find_packages

setup(name='lemol',
      version='0.0.2',
      description='Learning to Model Opponent Learning',
      url='https://github.com/ianrdavies/LeMOL',
      author='Ian Davies',
      author_email='ian.davies.12@ucl.ac.uk',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym@0.10.11', 'numpy-stl']
      )
