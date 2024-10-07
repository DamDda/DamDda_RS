from setuptools import setup, find_packages

setup(
    name='damdda_rs', 
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'run-server=damdda_rs.run:main',
        ],
    },
)