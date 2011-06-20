from setuptools import setup, find_packages

setup(
    name                 = "mlpack",
    version              = "0.1",
    packages             = find_packages(),
    include_package_data = True,
    scripts              = ["hmm.py", "maxent.py"],
    install_requires     = ["numpy>=1.6.0"],
    author               = "Rongzhou Shen",
    author_email         = "anticlockwise5@gmail.com",
    description          = "Package for implementation of many machine learning algorithms.",
    url                  = "https://github.com/anticlockwise/mlpack"
    )
