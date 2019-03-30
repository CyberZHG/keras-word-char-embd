import codecs
from setuptools import setup, find_packages


with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = reader.read()


with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))


setup(
    name='keras-word-char-embd',
    version='0.20.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-word-char-embd',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Concatenate word and character embeddings in Keras',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
