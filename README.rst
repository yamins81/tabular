Tabular
==========
Tabular data can be easily represented in Python using the language's native objects -- e.g. by lists of tuples representing the records of the data set.    Though easy to create, these kind of representations typically do not enable important tabular data manipulations, like efficient column selection, matrix mathematics, or spreadsheet-style operations. 

**Tabular** is a package of Python modules for working with tabular data.     Its main object is the **tabarray** class, a data structure for holding and manipulating tabular data.  By putting data into a **tabarray** object, you'll get a representation of the data that is more flexible and powerful than a native Python representation.   More specifically, **tabarray** provides:
	
*	ultra-fast filtering, selection, and numerical analysis methods, using convenient Matlab-style matrix operation syntax
*	spreadsheet-style operations, including row & column operations, 'sort', 'replace',  'aggregate', 'pivot', and 'join'
*	flexible load and save methods for a variety of file formats, including delimited text (CSV), binary, and HTML
*	sophisticated inference algorithms for determining formatting parameters and data types of input files
*	support for hierarchical groupings of columns, both as data structures and file formats

**Note to NumPy Users:**  The **tabarray** object is based on the `ndarray <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html?highlight=ndarray#numpy.ndarray>`_ object from the Numerical Python package (`NumPy <http://numpy.scipy.org/>`_), and the Tabular package is built to interface well with NumPy in general.  In particular, users of NumPy can get many of the benefits of Tabular, e.g. the spreadsheet-style operations, without having replace their usual NumPy objects with tabarrays, since most of the useful functional pieces of Tabular are written to work directly on NumPy ndarrays and record arrays (see `relationship to NumPy <http://web.mit.edu/yamins/www/tabular/reference/organization.html#relation-to-numpy>`_).


Download
----------------------------

Download the latest release of tabular from the Python Package Index (PyPi):  http://pypi.python.org/pypi/tabular/.    

Tabular requires Python 2.6 or higher, but will not work with Python 3k (since NumPy itself is not ported to Py3k).  Tabular **requires** NumPy v1.6 or higher.  Any earlier version WILL NOT WORK.

Once these dependencies are installed, you can simply go to the Tabular source directory in your terminal and run the command "python setup.py install" (see `Installing Python Modules <http://docs.python.org/install/index.html>`_).

You can also clone our github repository: https://github.com/yamins81/tabular.   You can report bugs, make suggestions, submit pull requests, and follow an RSS from our github site.  


Documentation
--------------------------------
http://web.mit.edu/yamins/www/tabular/
