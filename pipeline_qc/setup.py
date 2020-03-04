import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pipeline-qc",
    version="0.0.1",
    authors="Aditya Nath, Calysta Yan",
    author_email="aditya.nath@alleninstitute.org",
    description="Contains all the automatic image qc fns for use with pipeline images",
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

)
