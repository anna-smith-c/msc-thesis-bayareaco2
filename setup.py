from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="bayareaco2",
    version="1.0",
    description="Bay Area CO2 Explorer",
    author="Anna C. Smith",
    packages=find_packages(),
    install_requires=requirements,
)