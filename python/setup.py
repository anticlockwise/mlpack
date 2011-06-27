from setuptools import setup, find_packages, Extension
from mlpack import get_version
from glob import glob

lib_sources = glob("svm_light/*.c")
lib_sources.remove("svm_light/svm_loqo.c")
lib_sources.remove("svm_light/svm_classify.c")

setup(
    name                 = "mlpack",
    version              = get_version(),
    packages             = find_packages(),
    include_package_data = True,
    scripts              = ["hmm.py", "maxent.py"],
    install_requires     = ["numpy>=1.6.0"],
    author               = "Rongzhou Shen",
    author_email         = "anticlockwise5@gmail.com",
    description          = "Package for implementation of many machine learning algorithms.",
    url                  = "https://github.com/anticlockwise/mlpack",
	ext_modules = [
		Extension("svmlight", include_dirs=["svm_light"],
			sources = ["mlpack/svm/svm_light_wrapper.c"] + lib_sources)
		])
