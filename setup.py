"""
simab - Simple Multi-Armed Bandit Simulator
-------------------------------------------

"""
import re
import ast
from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('simab/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
        name='simab',
        version=version,
        description='Simple Multi-Armed Bandit Simulator',
        long_description=__doc__,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Internet :: WWW/HTTP :: Dynamic Content :: CGI Tools/Libraries',
            'Topic :: Utilities',
            'License :: OSI Approved :: MIT License',
            ],
        keywords='MAB, Multi-Armed Bandit, Machine Learning, Reinforcement Learning',
        author='Akihiko ITOH',
        author_email='itoh.akihiko.5@facebook.com',
        url='https://github.com/AkihikoITOH/simab',
        license='MIT',
        packages=find_packages(exclude=['examples', 'tests', 'locals']),
        include_package_data=True,
        zip_safe=True,
        # long_description=read_md('README.md'),
        install_requires=['numpy', 'scikit-learn'],
    )

