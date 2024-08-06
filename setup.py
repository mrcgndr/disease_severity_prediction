from setuptools import find_packages, setup

from disease_pred import __author__, __author_email__, __copyright__, __docs__, __homepage__, __license__, __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="disease_severity_prediction",
    version=__version__,
    description=__docs__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__author_email__,
    url=__homepage__,
    download_url=__homepage__,
    license=__license__,
    copyright=__copyright__,
    keywords=["remote-sensing", "agriculture", "image-processing", "machine-learning"],
    packages=find_packages(),
    python_requires=">=3.11",
)
