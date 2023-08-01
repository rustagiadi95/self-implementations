from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requires = fh.read().split("\n")
    requires = [r.strip() for r in requires if len(r.strip()) > 0]
    requires = [r for r in requires if "${" not in r]

setup(
    name='self_implementations',
    packages=find_packages(),
    version='0.1.0',
    description='Module for self implememntations of research papers',
    author='rustagiadi95',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={"Bug Tracker": ""},
    install_requires=requires,
)
