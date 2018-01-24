from setuptools import setup, find_packages
setup(
    name='pystats',
    version='0.0.1',
    description='Python package for data analysis.',
    long_description='README.md',
    author='Yoshimasa Sakuragi',
    author_email='ysakuragi16@gmail.com',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib'],
    url='https://github.com/SakuragiYoshimasa/pyplfv',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
