Requirements
------------

- python=3.7
- root>=6.20
- numpy
- matplotlib
- tensorflow==2.2.0
- scikit-learn==0.23.1

Operative system
''''''''''''''''

Supported:

- Linux
- MacOS (not tested)

This project uses the package PyROOT which is currently not supported
on Windows. But there is a workaround, jump to `Colab way`_.

Conda environment
'''''''''''''''''

The best option is to set a conda virtual environment.
Go to this link and follow instruction to install conda:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

A conda configuration file to create an environment that meets all
the requirements can be found on the project repository
`here <https://github.com/sbenegiano/cmepda/blob/master/cmepda_H2e2mu.yml>`_.

Once conda is installed a virtual environment can be set up
just executing::

    $ conda env create --file cmepda_H2e2mu.yml

then activate the new virtual environment::

    $ conda activate cmepda_H2e2mu

and check if everything is ok::

    $ python
    >>> import ROOT
    >>> import numpy
    >>> import matplotlib
    >>> import tensorflow
    >>> import sklearn
    >>> exit()

if no import error are raised the environment is ready to run
the project module(s).

Colab way
'''''''''

If previous steps went wrong, OS is not supported or simply there is no
will to run the project locally an alternative is proposed.
A Colab notebook has been prepared to set the environment to run the
project.

Visit this
`link <https://colab.research.google.com/drive/1uqhEn-AOCnOiT0L6UO1IP9tX8pw2f3m2?usp=sharing>`_
and follow all the steps to get the code running.

.. Root
.. ''''
.. Setting up Root can be tricky, to make things easy use conda.

.. https://anaconda.org/conda-forge/root



