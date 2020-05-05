import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Python optimization workflow", # Replace with your own username
    version="0.0.1",
    author="Jingxiang Guo",
    author_email="jguo10@uh.edu",
    description="Systematic and Reproduciable parameterization of force-field",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    entry_points={ 
            'console_scripts':[ 
                "optimize=main.__main__:main",
                "clearjob=main.__main__:main"

            ] 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

