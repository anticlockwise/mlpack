Tutorial
============

This documentation is meant to give you an overview of the MLPack structure
and the basic usage of the scripts and code.

Installation
-------------

MLPack is a standard Python package, so it can be installed using the following
command ::

   $ python setup.py install

On some systems, you will need to install as the root user.

Using the scripts
------------------

Once installed, each learning algorithm will provide a command line script to be invoked
against an input file. Input files can be supervised-learning feature files,
observations sequences (Hidden Markov Model) or a mini descriptive language.
Each script will take configuration parameters for the learning algorithm
through the form of a configuration file (CFG/INI files).

For example, to train a Maximum Entropy Model ::

   $ maxent.py -t -m sample_model sample_data.txt

The ``-m`` option specified the file that the model should be written to in
this instance. The ``-t`` option specifies that you're executing the script
for training instead of predicting, if it's unspecified, the script will be used for
prediction based on the model read in by the ``-m`` option.

See :doc:`scripts` for a full reference on the commandline scripts, including
learning algorithm configuration options and input file formats.

Using the code
---------------

If you're already familiar with Python, then using the code is very straight forward.
To programmatically use a particular algorithm, please see :doc:`extend` for more
details.
