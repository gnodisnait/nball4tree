from setuptools import setup, find_packages

setup(name='nball4tree',
      version='0.1',
      description='nball for tree embedding',
      url='http://github.com/gnodisnait/nball4tree',
      author='gnod isnait',
      author_email='gnodisnait@gmail.com',
      license='MIT',
      packages= find_packages(),
      package_data = {'': ['*.pickle'],},
      zip_safe=False)
