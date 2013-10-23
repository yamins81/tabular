'''
Class and functions pertaining to the tabular.tabarray class.

The :class:`tabarray` class is a column-oriented hierarchical data object and 
subclass of `numpy.ndarray <http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html?highlight=ndarray#numpy.ndarray>`_.

The basic structure of this module is that it contains:

*	The tabarray class.

*	Some helper functions for tabarray.  The helper functions are precisely 
	those necessary to wrap functions from the :mod:`tabular.spreadsheet` 
	module that operate on lists of arrays, to handle tabular's additional 
	structure.  These functions are named with the convention "tab_FNAME", e.g. 
	"tab_rowstack", "tab_join" &c.  The functions in :mod:`tabular.spreadsheet` 
	that only take a single array are all wrapped JUST as methods of tabarray, 
	and not as separate functions.

'''

import os
import csv

import numpy as np

import tabular.io as io
import tabular.spreadsheet as spreadsheet
import tabular.utils as utils

__all__ = ['tabarray', 'tab_colstack', 'tab_rowstack','tab_join']

DEFAULT_VERBOSITY=io.DEFAULT_VERBOSITY

def modifydocs(a, b, desc=''):
    """
    Convenience function for writing documentation.

    For a class method `a` that is essentially a wrapper for an outside 
    function `b`, rope in the docstring from `b` and append to that of `a`.  
    Also modify the docstring of `a` to get the indentation right.
    
    Will probably deprecate this soon.

    **Parameters**

		**a** :  class method

			Class method wrapping `b`.

		**b** :  function

			Function wrapped by `a`.

		**desc** :  string, optional

			Description of `b`, e.g. restructured text providing a link to the 
			documentation for `b`.  Default is an empty string.

    **Returns**

		**newdoc** :  string

			New docstring for `a`.

    """
    newdoc = a.func_doc.replace('\t\t', '\t')
    newdoc += "Documentation from " + desc + ":\n" + b.func_doc
    return newdoc

def tab_colstack(ListOfTabArrays, mode='abort'):
    """
    "Horizontal stacking" of tabarrays, e.g. adding columns.

    Wrapper for :func:`tabular.spreadsheet.colstack` that deals with the 
    coloring and returns the result as a tabarray.

    Method calls::

        data = tabular.spreadsheet.colstack(ListOfTabArrays, mode=mode)

    """
    (data, naming) = spreadsheet.colstack(ListOfTabArrays, mode=mode, 
                                          returnnaming=True)
        
    coloring = {}
    for (i, a) in enumerate(ListOfTabArrays):
        namedict = dict([(x,y) for (j,x,y) in naming if i == j])
        for k in a.coloring:
            s = [namedict[kk] for kk in a.coloring[k]]
            if k in coloring.keys():
                coloring[k] = utils.uniqify(coloring[k] + s)
            else:
                coloring[k] = s

    for k in coloring.keys():
        s = [x for x in coloring[k] if x in data.dtype.names]
        if len(s) > 0:
            coloring[k] = s
        else:
            coloring.pop(k)

    data = data.view(tabarray)
    data.coloring = coloring
    return data
tab_colstack.func_doc = modifydocs(tab_colstack, spreadsheet.colstack, 
                                   ":func:`tabular.spreadsheet.colstack`")

def tab_rowstack(ListOfTabArrays, mode='nulls'):
    """
    "Vertical stacking" of tabarrays, e.g. adding rows.

    Wrapper for :func:`tabular.spreadsheet.rowstack` that deals with the 
    coloring and returns the result as a tabarray.

    Method calls::

        data = tabular.spreadsheet.rowstack(ListOfTabArrays, mode=mode)

    """
    data = spreadsheet.rowstack(ListOfTabArrays, mode=mode)

    coloring = {}
    for a in ListOfTabArrays:
        for k in a.coloring:
            if k in coloring.keys():
                coloring[k] = utils.uniqify(coloring[k] + a.coloring[k])
            else:
                coloring[k] = a.coloring[k]
    for k in coloring.keys():
        s = [x for x in coloring[k] if x in data.dtype.names]
        if len(s) > 0:
            coloring[k] = s
        else:
            coloring.pop(k)

    data = data.view(tabarray)
    data.coloring = coloring
    return data
tab_rowstack.func_doc = modifydocs(tab_rowstack, spreadsheet.rowstack, 
                                   ":func:`tabular.spreadsheet.rowstack`")

def tab_join(ToMerge, keycols=None, nullvals=None, renamer=None, 
             returnrenaming=False, Names=None):
    '''
    Database-join for tabular arrays.

    Wrapper for :func:`tabular.spreadsheet.join` that deals with the coloring 
    and returns the result as a tabarray.

    Method calls::

            data = tabular.spreadsheet.join

    '''

    [Result,Renaming] = spreadsheet.join(ToMerge, keycols=keycols, 
          nullvals=nullvals, renamer=renamer, returnrenaming=True, Names=Names)

    if isinstance(ToMerge,dict):
        Names = ToMerge.keys()
    else:
        Names = range(len(ToMerge))

    Colorings = dict([(k,ToMerge[k].coloring) if 'coloring' in dir(ToMerge[k])  
                                              else {} for k in Names])
    for k in Names:
        if k in Renaming.keys():
            l = ToMerge[k]
            Colorings[k] = \
                dict([(g, [n if not n in Renaming[k].keys() else Renaming[k][n] 
                       for n in l.coloring[g]]) for g in Colorings[k].keys()])
    Coloring = {}
    for k in Colorings.keys():
        for j in Colorings[k].keys():
            if j in Coloring.keys():
                Coloring[j] = utils.uniqify(Coloring[j] + Colorings[k][j])
            else:
                Coloring[j] = utils.uniqify(Colorings[k][j])

    Result = Result.view(tabarray)
    Result.coloring = Coloring

    if returnrenaming:
        return [Result,Renaming]
    else:
        return Result


class tabarray(np.ndarray):
    """
    Subclass of the numpy ndarray with extra structure and functionality.

    tabarray is a column-oriented data object based on the numpy ndarray with
    structured dtype, with added functionality and ability to define named 
    groups of columns.

    tabarray supports several i/o methods to/from a number of file formats, 
    including (separated variable) text (e.g. ``.txt``, ``.tsv``, ``.csv``)
    and numpy binary (``.npz``).

    Added functionality includes spreadsheet style operations such as "pivot", 
    "aggregate" and "replace".

    See docstring of the tabarray.__new__ method, or the Tabular reference documentation, for data on constructing a tabarrays. 
    
    """

    def __new__(subtype, array=None, records=None, columns=None, SVfile=None, 
                binary=None, shape=None, 
                dtype=None, formats=None, names=None, titles=None, 
                aligned=False, byteorder=None, buf=None, offset = 0,
                strides = None, comments=None, delimiter=None, 
                lineterminator='\n', escapechar=None, quoting=None, 
                quotechar=None, doublequote=True, skipinitialspace=False,
                skiprows=0, uselines=None, usecols=None, excludecols=None,
                toload=None, metametadata=None, kvpairs=None,
                namesinheader=True, headerlines=None, valuefixer=None, 
                linefixer=None, colfixer = None, delimiter_regex = None, coloring=None, inflines=2500, wrap=None, 
                typer = None, missingvalues = None, fillingvalues = None, renamer = None,verbosity=DEFAULT_VERBOSITY):
        """
        Unified constructor for tabarrays.

        **Specifying the data:**

                Data can be passed to the constructor, or loaded from several 
                different file formats. 

                **array** :  two-dimensional arrays (:class:`numpy.ndarray`)


                        >>> import numpy
                        >>> x = numpy.array([[1, 2], [3, 4]])
                        >>> tabarray(array=x)
                        tabarray([(1, 2), (3, 4)], 
                              dtype=[('f0', '<i4'), ('f1', '<i4')])
                        
                        **See also:**  `numpy.rec.fromrecords <http://docs.scipy.org/doc/numpy/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords>`_

                **records** :  python list of records (elemets can be tuples or lists)
 
                        >>> tabarray(records=[('bork', 1, 3.5), ('stork', 2, -4.0)], names=['x','y','z'])
                        tabarray([('bork', 1, 3.5), ('stork', 2, -4.0)], 
                              dtype=[('x', '|S5'), ('y', '<i4'), ('z', '<f8')])

                        **See also:**  `numpy.rec.fromrecords <http://docs.scipy.org/doc/numpy/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromrecords>`_


                **columns** :  list of python lists or 1-D numpy arrays 
                
                        Fastest when passed a list of numpy arrays, rather than
                        a list of lists.

                        >>> tabarray(columns=[['bork', 'stork'], [1, 2], [3.5, -4.0]], names=['x','y','z']) 
                        tabarray([('bork', 1, 3.5), ('stork', 2, -4.0)], 
                              dtype=[('x', '|S5'), ('y', '<i4'), ('z', '<f8')])


                **kvpairs** : list of list of key-value pairs

                        For loading key-value pairs (e.g. as from an XML file).    Missing values can be specified using the **fillingvalues** argument.  

                **See also:**  `numpy.rec.fromarrays <http://docs.scipy.org/doc/numpy/reference/generated/numpy.core.records.fromrecords.html#numpy.core.records.fromarrays>`_
                

                **SVfile** :  string

                        File path to a separated variable (CSV) text file.  
                        Load data from a CSV by calling::

                                tabular.io.loadSV(SVfile, comments, delimiter, 
                                lineterminator, skiprows, usecols, metametadata, 
                                namesinheader, valuefixer, linefixer)

                        **See also:**  :func:`saveSV`, 
                        :func:`tabular.io.loadSV`


                **binary** :  string

                        File path to a binary file. Load a ``.npz`` binary file 
                        created by the :func:`savebinary` by calling::

                                tabular.io.loadbinary(binary)

                        which uses :func:`numpy.load`.

                        **See also:** :func:`savebinary`, 
                        :func:`tabular.io.loadbinary`

        **Additional parameters:**
                
                **names** : list of strings
                	
                	Sets the names of the columns of the resulting tabarray.   If not specified, `names` value is determined first by looking for metadata in the header of the file, and if that is not found, are assigned by NumPy's `f0, f1, ... fn` convention.     See **namesinheader** parameter below.
                	
                **formats** :  string or list of strings
                
                    Sets the datatypes of the columns.  The value of `formats` can be a list or comma-delimited string of values describing values for each column (e.g. "str,str,int,float" or ["str", "str", "int", "float"]), a single value to apply to all columns, or anything that can be used in numpy.rec.array constructor.   
                    
                    If the **formats** (or **dtype**) parameter are not  specified, typing is done by inference.   (See also **typer** parameter below).  
                    
                        
                **dtype** : numpy dtype object
                
                    Sets the numpy dtype of the resulting tabarray, combining 
                    column format and column name information.  If dtype is set, any **names** and **formats** specifications will be overriden.   If the **dtype** (or **formats**) parameter are not  specified, typing is done by inference.   (See also **typer** parameter below).   
  
                The **names**, **formats** and **dtype** parameters duplicate parameters of the NumPy record array creation inferface.   Additional paramters of the NumPy inferface that are passed through are **shape**, **titles**, **byteorder** and **aligned** (see NumPy documentation for more information.)
                
             
                **delimiter** : single-character string
                
                    When reading text file, character to use as delimiter to split fields.  If not specified, the delimiter is determined first by looking for special-format metadata specifying the delimiter, and then if no specification is found, attempts are made to infer delimiter from file contents.   (See **inflines** parameter below.)  
                    
                **delimiter_regex** :  regular expression (compiled or in string format)                    
                 
                    Regular expression to use to recognize delimiters, in place of a single character.   (For instance, to have whitespace delimiting, using delimiter_regex = '[\s*]+' )
                                 
                **lineterminator** : single-character string
                
                    Line terminator to use when reading in using SVfile
                    
                **skipinitialspace** : boolean
                    If true, strips whitespace following the delimiter from field.   
                    
               The **delimiter**, **linterminator** and **skipinitialspace** 
               parameters are passed on as parameters to the python CSV module, 
               which is used for reading in delimited text files.   Additional 
               parameters from that interface that are replicated in this constructor include **quotechar**, **escapechar**, **quoting**, **doublequote** and **dialect** (see CSV module documentation for more information.)

                **skiprows** :  non-negative integer, optional

                    When reading from a text file, the first `skiprows` lines are ignored.  Default is 0, e.g no rows are skipped. 

                **uselines** : pair of non-negative integer, optional
                
                    When reading from a text file, range of lines of data to load.  (In constrast to **skiprows**, which specifies file rows to ignore before looking for header information, **uselines** specifies which data (non-header) lines to use, after header has been striped and processed.)   See **headerlines** below.

                **usecols** :  sequence of non-negative integers or strings, optional

                    When reading from a text file, only the columns in *usecols* are loaded and processed.   Columns can be described by number, with 0 being the first column; or if name metadata is present, then by name ; or, if color group information is present in the file, then by color group name.   (Default is None, e.g. all columns are loaded.)

                **excludecols** :  sequence of non-negative integers or strings, optional

                    Converse of **usecols**, e.g. all columns EXCEPT those listed will be loaded. 
                    
                **comments** : single-character string, optional
                    
                    When reading from a text file, character used to distinguish header lines.  If specified, any lines beginning with this character at the top of the file are assumed to contain header information and not row data. 
              
                **headerlines** : integer, optional

                    When reading from a text file, the number of lines at the top of the file (after the first  `skiprows` lines) corresponding to the header of the file, where metadata can be found.   Lines after headerlines are assumed to contain row contents.   If not specified, value is determined first by looking for special metametadata  in first line of file (see Tabular reference documentation for more information about this), and if no such metadata is found, is inferred by looking at file contents.    
                    
                **namesinheader** : Boolean, optional

                    When reading from a text file, if `namesinheader == True`, then assume the column names are in the last header line (unless overridden by existing metadata or metametadata directive).    Default is True.                        
                    
                **linefixer** : callable, optional

                   When reading from a text file, this callable is applied to every line in the file.  This option is passed on all the way to the call to `io.loadSVrecord` function, and is applied directly to the strings in the file, after they're split in lines but before they're split into fields or any typing is done.   The purpose is to make lines with errors or mistakes amenable to delimiter inference and field-splitting. 
                    
                **valuefixer**  :  callable, or list or dictionary of callables, optional

                   When reading from a text file, these callable(s) are applied to every value in each field.   The application is done after line strings are loaded and split into fields, but before any typing or missing-value imputation is done.  The purpose of the **valuefixer** is to prepare column values for typing and imputation.   The valuefixer callable can return a string or a python object.   If `valuefixer` is a single callable, then that same callable is applied to values in all column; if it is a dictionary, then the keys can be either numbers or names and the value for the key will be applied to values in the corresponding column with that name or number; if it is a list, then the list elements must be in 1-1 correponsdence with the loaded columns, and are applied to each respectively.
                   
                **colfixer** : callable, or list or dictionary of callables, optional

                    Same as **valuefixer**,  but instead of being applied to individual values, are applied to whole columns (and must return columns or numpy arrays of identical length).    Like valuefixer, colfixer callable(s) are applied before typing and missing-value imputation.  
                    
                **missingvalues** : string, callable returning string, or list or dictionary of strings or string-valued callable
                
                    When reading from text file, string value to consider as "missing data" and to be replaced before typing is done.   If specified as a callable, the callable will be applied to the column(s) to determine missing value.   If specified as a dictionary, keys are expected to be numbers of names of columns, and values are individual missing values for those columns (like **valuefixer** inferface).   
                    
                    
                **fillingvalues** : string, pair of strings, callable returning string, or list or dictionary of strings or string-valued callable
                
                    When reading from text file, values to be used to replace missing data before typing is done.   If specified as a  single non-callable, non-tuple value, this value is used to replace all missing data.  If specified as a callable, the callable is applied to the column and returns the fill value (e.g. to allow the value to depend on the column type).    If specified as a pair of values, the first value acts as the missing value and the second as the value to replace with.   If a dictionary or list of values, then values are applied to corresponding columns.  
 
                NOTE:  all the **missingvalues** and **fillingvalues** functionalities can be replicated (and generalized) using the **valuefixer** or **colfixer** parameters, by specifying function(s) which identify and replace missing values.   While more limited, using **missingvalues** and **fillingvalues**  interface is easier and gives better performance.   
 
                **typer**  :   callable taking python list of strings (or other values) and returning 1-dnumpy array ; or list dictionary of such callables  
                
                   Function used to infer type and convert string lists into typed numpy arrays, if no format information has been provided.   When applied at all, this function is applied after string have been loaded and split into fields.   This function is expected to impute missing values as well, and will override any setting of **missingvalues** or **fillingvalues**.    If a callable is passed,  it is used as typer for all columns, while if a dictionary (or list) of callables is passed, they're used on corresponding columns.    If needed (e.g. because formatting information hasn't been supplied) but **typer** isn't specified (at least, for a given column), the constructor defaults to using the `utils.DEFAULT_TYPEINFERER` function.      


                **inflines** :  integer, optional
                
                    Number of lines of file to use as sample data when inferring delimiter and header.   
  
                **metametadata** :  dictionary of integers or pairs of integers
                    
                    Specifies supplementary metametadata information for use 
                    with SVfile loading.  See Tabular reference documentation for more information
                                      
                **coloring**:  dictionary

                        Hierarchical column-oriented structure.  

                   *	Colorings can be passed as argument:

                        *	In the *coloring* argument, pass a dictionary. Each 
                        	key is a string naming a color whose corresponding
                        	value is a list of column names (strings) in that 
                        	color.

                        *	If colorings are passed as argument, they override
                        	any colorings inferred from the input data.

                   *	Colorings can be inferred from the input data:

                        *	If constructing from a CSV file (e.g. ``.tsv``, 
                        	``.csv``) created by :func:`saveSV`, colorings are 
                        	automatically parsed from the header when present.

                        *	If constructing from a numpy binary file (e.g. 
                        	``.npz``) created by :func:`savebinary`, colorings 
                        	are automatically loaded from a binary file 
                        	(``coloring.npy``) in the ``.npz`` directory.

                **wrap**:  string

                        Adds a color with name  *wrap* listing all column 
                        names. 

                **verbosity** :  integer, optional

                   Sets how much detail from messages will be printed.

     **Special column names:**

                        Column names that begin and end with double
                        underscores, e.g. '__column_name__' are used
                        to hold row-by-row metadata and specify  arbitrary higher-level groups of rows, in analogy to how the `coloring` attribute specifies groupings of columns.

                        One use of this is for formatting
                        and communicating "side" information to other
                        :class:`tabarray` methods.  For instance:

                        *	A '__color__' column is interpreted by the 
                            tabular.web.tabular2html function to specify row color in making html representations of tabarrays.   It is expected in each row to contain a web-safe hex triplet color specification, e.g. a string of the form '#XXXXXX' (see  http://en.wikipedia.org/wiki/Web_colors).

                        *	The '__aggregates__' column is used to disambiguate
                        	rows that are aggregates of data in other sets of
                        	rows for the ``.aggregate_in`` method (see comments 
                        	on that method).
       
        """
        metadata = {}
        if not array is None:
            if len(array) > 0:
                DataObj = utils.fromrecords(array, type=np.ndarray, dtype=dtype, 
                          shape=shape, formats=formats, names=names, 
                          titles=titles, aligned=aligned, byteorder=byteorder)
            else:
                DataObj = utils.fromarrays([[]]*len(array.dtype), type=np.ndarray,
                          dtype=dtype, shape=shape, formats=formats, 
                          names=names, titles=titles, aligned=aligned, 
                          byteorder=byteorder)
        elif not records is None:
            DataObj = utils.fromrecords(records, type=np.ndarray, dtype=dtype, shape=shape, 
                      formats=formats, names=names, titles=titles, 
                      aligned=aligned, byteorder=byteorder)
        elif not columns is None:
            DataObj = utils.fromarrays(columns,type=np.ndarray, dtype=dtype,
                      shape=shape, formats=formats, names=names, titles=titles, 
                      aligned=aligned, byteorder=byteorder)
        elif not kvpairs is None:
        	DataObj = utils.fromkeypairs(kvpairs,dtype=dtype,fillingvalues=fillingvalues)
        elif not SVfile is None:
            chkExists(SVfile)
            # The returned DataObj is a list of numpy arrays.
            [DataObj, metadata] = \
                io.loadSV(fname=SVfile, names=names, dtype=dtype, shape=shape, 
                formats=formats, titles=titles, aligned=aligned, 
                byteorder=byteorder, buf=buf,strides=strides,
                comments=comments, delimiter=delimiter, 
                lineterminator=lineterminator, escapechar=escapechar,
                quoting=quoting,quotechar=quotechar,doublequote=doublequote,
                skipinitialspace=skipinitialspace, skiprows=skiprows, 
                uselines=uselines, usecols=usecols, excludecols=excludecols, 
                metametadata=metametadata, namesinheader=namesinheader, 
                headerlines=headerlines, valuefixer=valuefixer, colfixer=colfixer, linefixer = linefixer, missingvalues = missingvalues, fillingvalues=fillingvalues, typer = typer, delimiter_regex = delimiter_regex, inflines=inflines, renamer=renamer,verbosity=verbosity)
            if (names is None) and 'names' in metadata.keys() and metadata['names']:
                names = metadata['names']
            if (coloring is None) and 'coloring' in metadata.keys() and metadata['coloring']:
                coloring = metadata['coloring']
            

        elif not binary is None:
            chkExists(binary)
            # Returned DataObj is a numpy ndarray with structured dtype
            [DataObj, givendtype, givencoloring] = io.loadbinary(fname=binary)
            if (dtype is None) and (not givendtype is None):
                dtype = givendtype
            if (coloring is None) and (not givencoloring is None):
                coloring = givencoloring
            DataObj = utils.fromrecords(DataObj, type=np.ndarray, dtype=dtype, shape=shape,   
                      formats=formats, names=names, titles=titles, 
                      aligned=aligned, byteorder=byteorder)
        else:
            DataObj = np.core.records.recarray.__new__(
                      subtype, shape, dtype=dtype, 
                      formats=formats, names=names, titles=titles, 
                      aligned=aligned, byteorder=byteorder, buf=buf, 
                      offset=offset, strides=strides)
                      
        DataObj = DataObj.view(subtype)
        
        DataObj.metadata = metadata

        if not coloring is None:
            coloringsInNames = \
                 list(set(coloring.keys()).intersection(set(DataObj.dtype.names)))
            if len(coloringsInNames) == 0:
                DataObj.coloring = coloring
            else:
                print ("Warning:  the following coloring keys,", 
                       coloringsInNames, ", are also attribute (column) names " 
                       "in the tabarray.  This is not allowed, and so these " 
                       "coloring keys will be deleted.  The corresponding "
                       "columns of data will not be lost and will retain the "
                       "same names.")
                for c in coloringsInNames:
                    coloring.pop(c)
                DataObj.coloring = coloring
        else:
            DataObj.coloring = {}

        if not wrap is None:
            DataObj.coloring[wrap] = DataObj.dtype.names

        return DataObj
        
    def __array_finalize__(self, obj):
        """
        Set default attributes (e.g. `coloring`) if `obj` does not have them.

        Note:  this is called when you view a numpy ndarray as a tabarray.

        """
        self.coloring = getattr(obj, 'coloring', {})
        
    def __array_wrap__(self, arr, context=None):
        if arr.ndim == 0:
            pass
        else:
            return arr

    def extract(self):
        """
        Creates a copy of this tabarray in the form of a numpy ndarray.

        Useful if you want to do math on array elements, e.g. if you have a 
        subset of the columns that are all numerical, you can construct a 
        numerical matrix and do matrix operations.

        """
        return np.vstack([self[r] for r in self.dtype.names]).T.squeeze()

    def __getitem__(self, ind):
        """
        Returns a subrectangle of the table.

        The representation of the subrectangle depends on `type(ind)`. Also, 
        whether the returned object represents a new independent copy of the 
        subrectangle, or a "view" into this self object, depends on 
        `type(ind)`.

        *	If you pass the name of an existing coloring, you get a tabarray 
        	consisting of copies of columns in that coloring.

        *	If you pass a list of existing coloring names and/or column names, 
        	you get a tabarray consisting of copies of columns in the list 
        	(name of coloring is equivalent to list of names of columns in that 
        	coloring; duplicate columns are deleted).

        *	If you pass a :class:`numpy.ndarray`, you get a tabarray consisting 
        	a subrectangle of the tabarray, as handled by  
        	:func:`numpy.ndarray.__getitem__`:

                *	if you pass a 1D NumPy ndarray of booleans of `len(self)`,    
                	the rectangle contains copies of the rows for which the 
                	corresponding entry is `True`.

                *	if you pass a list of row numbers, you get a tabarray
                	containing copies of these rows.

        """
        if ind in self.coloring.keys():
            return self[self.coloring[ind]]
        elif isinstance(ind,list) and self.dtype.names and \
             all([a in self.dtype.names or a in self.coloring.keys() 
                                                           for a in ind]) and \
             set(self.coloring.keys()).intersection(ind):
            ns = utils.uniqify(utils.listunion([[a] if a in self.dtype.names 
                                          else self.coloring[a] for a in ind]))
            return self[ns]
        else:
            D = np.ndarray.__getitem__(self, ind)
            if isinstance(D, np.ndarray) and not (D.dtype.names is None):
                D = D.view(tabarray)
                D.coloring = dict([(k, 
                list(set(self.coloring[k]).intersection(set(D.dtype.names)))) 
                for k in self.coloring.keys() if 
                len(set(self.coloring[k]).intersection(set(D.dtype.names))) > 0 ])
            return D

    def addrecords(self, new):
        """
        Append one or more records to the end of the array.

        Method wraps::

                tabular.spreadsheet.addrecords(self, new)

        """
        data = spreadsheet.addrecords(self,new)
        data = data.view(tabarray)
        data.coloring = self.coloring
        return data
    addrecords.func_doc = modifydocs(addrecords, spreadsheet.addrecords, 
                                     ":func:`tabular.spreadsheet.addrecords`")

    def addcols(self, cols, names=None):
        """
        Add one or more new columns.

        Method wraps::

                tabular.spreadsheet.addcols(self, cols, names)

        """
        data = spreadsheet.addcols(self, cols, names)
        data = data.view(tabarray)
        data.coloring = self.coloring
        return data
    addcols.func_doc = modifydocs(addcols, spreadsheet.addcols, 
                                  ":func:`tabular.spreadsheet.addcols`")

    def deletecols(self, cols):
        """
        Delete columns and/or colors.

        Method wraps::

                tabular.spreadsheet.deletecols(self, cols)

        """
        if isinstance(cols, str):
        	cols = cols.split(',')
        deletenames = utils.uniqify(utils.listunion([[c] if c in 
        self.dtype.names else self.coloring[c] for c in cols]))
        return spreadsheet.deletecols(self,deletenames)
    deletecols.func_doc = modifydocs(deletecols, spreadsheet.deletecols, 
                                     ":func:`tabular.spreadsheet.deletecols`")

    def renamecol(self, old, new):
        """
        Rename column or color in-place.

        Method wraps::

                tabular.spreadsheet.renamecol(self, old, new)

        """
        spreadsheet.renamecol(self,old,new)
        for x in self.coloring.keys():
            if old in self.coloring[x]:
                ind = self.coloring[x].index(old)
                self.coloring[x][ind] = new
    renamecol.func_doc = modifydocs(renamecol, spreadsheet.renamecol, 
                                    ":func:`tabular.spreadsheet.renamecol`")

    def saveSV(self, fname, comments=None, metadata=None, printmetadict=None,
                       dialect = None, delimiter=None, doublequote=True, 
                       lineterminator='\n', escapechar = None, quoting=csv.QUOTE_MINIMAL, 
                       quotechar='"', skipinitialspace=False, 
                       stringifier=None, verbosity=DEFAULT_VERBOSITY):
        """
        Save the tabarray to a single flat separated variable (CSV) text file.   
        
        Method wraps::

                tabular.io.saveSV.      
                
        See docstring of tabular.io.saveSV, or Tabular reference documentation,  for more information.        

        """
        io.saveSV(fname,self, comments, metadata, printmetadict, 
                        dialect, delimiter, doublequote, lineterminator, escapechar, quoting, quotechar,skipinitialspace,stringifier=stringifier,verbosity=verbosity)
                        
    saveSV.func_doc = modifydocs(saveSV, io.saveSV, 
                                 ":func:`tabular.io.saveSV`")

    def savebinary(self, fname, savecoloring=True):
        """
        Save the tabarray to a numpy binary archive (``.npz``).
        
        Save the tabarray to a ``.npz`` zipped file containing ``.npy`` binary 
        files for data, plus optionally coloring and/or rowdata or simply to a 
        ``.npy`` binary file containing the data but no coloring or rowdata.

        Method wraps::

                tabular.io.savebinary(fname, self, savecoloring, saverowdata)

        """
        io.savebinary(fname=fname, X=self, savecoloring=savecoloring)
    savebinary.func_doc = modifydocs(savebinary, io.savebinary, 
                                     ":func:`tabular.io.savebinary`")

    def colstack(self, new, mode='abort'):
        """
        Horizontal stacking for tabarrays.

        Stack tabarray(s) in `new` to the right of `self`.

        **See also**

                :func:`tabular.tabarray.tab_colstack`, 
                :func:`tabular.spreadsheet.colstack`

        """
        if isinstance(new,list):
            return tab_colstack([self] + new,mode)
        else:
            return tab_colstack([self, new], mode)

    colstack.func_doc = modifydocs(colstack, spreadsheet.colstack,  
                                   ":func:`tabular.spreadsheet.colstack`")

    def rowstack(self, new, mode='nulls'):
        """
        Vertical stacking for tabarrays.

        Stack tabarray(s) in `new` below `self`.

        **See also**

                :func:`tabular.tabarray.tab_rowstack`, 
                :func:`tabular.spreadsheet.rowstack`

        """
        if isinstance(new,list):
            return tab_rowstack([self] + new, mode)
        else:
            return tab_rowstack([self, new], mode)

    rowstack.func_doc = modifydocs(rowstack, spreadsheet.rowstack, 
                                   ":func:`tabular.spreadsheet.rowstack`")

    def aggregate(self, On=None, AggFuncDict=None, AggFunc=None, AggList =
                  None, returnsort=False,KeepOthers=True, keyfuncdict=None):
        """
        Aggregate a tabarray on columns for given functions.

        Method wraps::

                tabular.spreadsheet.aggregate(self, On, AggFuncDict, AggFunc, returnsort)

        """
        if returnsort:
            [data, s] = spreadsheet.aggregate(X=self, 
                     On=On, 
                     AggFuncDict=AggFuncDict, 
                     AggFunc=AggFunc, 
                     AggList=AggList, 
                     returnsort=returnsort, 
                     keyfuncdict=keyfuncdict)
        else:
            data = spreadsheet.aggregate(X=self, On=On, AggFuncDict=AggFuncDict, 
                     AggFunc=AggFunc, AggList = AggList, returnsort=returnsort, 
                     KeepOthers=KeepOthers,
                     keyfuncdict=keyfuncdict)
        data = data.view(tabarray)
        data.coloring = self.coloring
        if returnsort:
            return [data, s]
        else:
            return data
    aggregate.func_doc = modifydocs(aggregate, spreadsheet.aggregate, 
                                    ":func:`tabular.spreadsheet.aggregate`")

    def aggregate_in(self, On=None, AggFuncDict=None, AggFunc=None,
                 AggList=None, interspersed=True):
        """
        Aggregate a tabarray and include original data in the result.

        See the :func:`aggregate` method.

        Method wraps::

                tabular.summarize.aggregate_in(self, On, AggFuncDict, AggFunc, interspersed)

        """
        data = spreadsheet.aggregate_in(Data=self, On=On, 
               AggFuncDict=AggFuncDict, AggFunc=AggFunc, 
               AggList = AggList, interspersed=interspersed)
        data = data.view(tabarray)
        data.view = self.coloring
        return data

    aggregate_in.func_doc = modifydocs(aggregate_in, spreadsheet.aggregate_in,  
                                    ":func:`tabular.spreadsheet.aggregate_in`")

    def pivot(self, a, b, Keep=None, NullVals=None, order = None, prefix='_'):
        """
        Pivot with `a` as the row axis and `b` values as the column axis.

        Method wraps::

                tabular.spreadsheet.pivot(X, a, b, Keep)

        """
        [data,coloring] = spreadsheet.pivot(X=self, a=a, b=b, Keep=Keep, 
                          NullVals=NullVals, order=order, prefix=prefix)
        data = data.view(tabarray)
        data.coloring = coloring
        return data

    pivot.func_doc = modifydocs(pivot, spreadsheet.pivot, 
                                ":func:`tabular.spreadsheet.pivot`")

    def replace(self, old, new, strict=True, cols=None, rows=None):
    	"""
    	Replace `old` with `new` in the rows `rows` of columns `cols`.
    	
    	Method wraps::
    	
    	        tabular.spreadsheet.replace(self, old, new, strict, cols, rows)
    	
    	"""
        spreadsheet.replace(self, old, new, strict, cols, rows)

    replace.func_doc = modifydocs(replace, spreadsheet.replace,
                                  ":func:`tabular.spreadsheet.replace`")

    def join(self, ToMerge, keycols=None, nullvals=None, 
             renamer=None, returnrenaming=False, selfname=None, Names=None):
        """
        Wrapper for spreadsheet.join, but handles coloring attributes.

        The `selfname` argument allows naming of `self` to be used if `ToMerge` 
        is a dictionary.

        **See also:** :func:`tabular.spreadsheet.join`, :func:`tab_join`
        """

        if isinstance(ToMerge,np.ndarray):
            ToMerge = [ToMerge]

        if isinstance(ToMerge,dict):
            assert selfname not in ToMerge.keys(), \
             ('Can\'t use "', selfname + '" for name of one of the things to '  
              'merge, since it is the same name as the self object.')
            if selfname == None:
                try:
                    selfname = self.name
                except AttributeError:
                    selfname = 'self'
            ToMerge.update({selfname:self})
        else:
            ToMerge = [self] + ToMerge

        return tab_join(ToMerge, keycols=keycols, nullvals=nullvals, 
                   renamer=renamer, returnrenaming=returnrenaming, Names=Names)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        """
        Returns the indices that would sort an array.

        .. note::

                This method wraps `numpy.argsort`.  This documentation is 
                modified from that of `numpy.argsort`.

        Perform an indirect sort along the given axis using the algorithm 
        specified by the `kind` keyword.  It returns an array of indices of the 
        same shape as the original array that index data along the given axis 
        in sorted order.

        **Parameters**

                **axis** : int or None, optional

                        Axis along which to sort.  The default is -1 (the last 
                        axis). If `None`, the flattened array is used.

                **kind** : {'quicksort', 'mergesort', 'heapsort'}, optional

                        Sorting algorithm.

                **order** : list, optional

                        This argument specifies which fields to compare first, 
                        second, etc.  Not all fields need be specified.

        **Returns**

                **index_array** : ndarray, int

                        Array of indices that sort the tabarray along the 
                        specified axis.  In other words, ``a[index_array]`` 
                        yields a sorted `a`.

                **See Also**

                        sort : Describes sorting algorithms used.
                        lexsort : Indirect stable sort with multiple keys.
                        ndarray.sort : Inplace sort.

                **Notes**

                        See `numpy.sort` for notes on the different sorting 
                        algorithms.

                **Examples**

                        Sorting with keys:

                        >>> x = tabarray([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
                        >>> x
                        tabarray([(1, 0), (0, 1)], 
                              dtype=[('x', '<i4'), ('y', '<i4')])

                        >>> x.argsort(order=('x','y'))
                        array([1, 0])

                        >>> x.argsort(order=('y','x'))
                        array([0, 1])

        """
        index_array = np.core.fromnumeric._wrapit(self, 'argsort', axis, 
                                                     kind, order)
        index_array = index_array.view(np.ndarray)
        return index_array

def chkExists( path ):
    """If the given file or directory does not exist, raise an exception"""
    if not os.path.exists(path): 
        raise IOError("Directory or file %s does not exist" % path)
