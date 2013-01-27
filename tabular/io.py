"""
Functions for :class:`tabular.tab.tabarray` i/o methods, including to/from 
separated-value (CSV, e.g. ``.tsv``, ``.csv``) and other text files, as well as
binary files.

"""

import types
import csv
import cPickle
import os
import shutil
import compiler
import re
import tempfile
from compiler.ast import (Stmt, 
                          Tuple,
                          Assign,
                          AssName,
                          Dict,
                          Const,
                          List,
                          Discard,
                          Name)

import numpy as np
from numpy import int64

import tabular as tb
import tabular.utils as utils
from tabular.utils import uniqify, listunion, is_string_like

__all__ = ['loadSV', 'loadSVcols','loadSVrecs', 'saveSV', 'loadbinary',       
           'savebinary', 'inferdelimiterfromname', 'inferdialect', 
           'processmetadata', 'inferheader', 'readstoredmetadata']

DEFAULT_VERBOSITY = 5

def loadSV(fname, shape=None, titles=None, aligned=False, byteorder=None,  
           renamer=None, **kwargs):
    """
    Load a delimited text file to a numpy record array.

    Basically, this function calls loadSVcols and combines columns returned by 
    that function into a numpy ndarray with stuctured dtype.  Also uses and 
    returns metadata including column names, formats, coloring, &c. if these 
    items are determined during the loading process.

    **Parameters**

        **fname** :  string or file object

            Path (or file object) corresponding to a separated variable
            (CSV) text file.

         **names** : list of strings
                
            Sets the names of the columns of the resulting tabarray.   If 
            not specified, `names` value is determined first by looking for 
            metadata in the header of the file, and if that is not found, 
            are assigned by NumPy's `f0, f1, ... fn` convention.  See 
            **namesinheader** parameter below.
                
        **formats** :  string or list of strings
            
            Sets the datatypes of the columns.  The value of `formats` can 
            be a list or comma-delimited string of values describing values 
            for each column (e.g. "str,str,int,float" or 
            ["str", "str", "int", "float"]), a single value to apply to all 
            columns, or anything that can be used in numpy.rec.array 
            constructor.   
                
            If the **formats** (or **dtype**) parameter are not  specified, 
            typing is done by inference.  See **typer** parameter below.  
                                    
        **dtype** : numpy dtype object
             
            Sets the numpy dtype of the resulting tabarray, combining column 
            format and column name information.  If dtype is set, any 
            **names** and **formats** specifications will be overriden.  If 
            the **dtype** (or **formats**) parameter are not  specified, 
            typing is done by inference.  See **typer** parameter below.   

        The **names**, **formats** and **dtype** parameters duplicate 
        parameters of the NumPy record array creation inferface.  Additional 
        paramters of the NumPy inferface that are passed through are 
        **shape**, **titles**, **byteorder** and **aligned** (see NumPy 
        documentation for more information.)

    **kwargs**: keyword argument dictionary of variable length

        Contains various parameters to be passed down to loadSVcols.  These may 
        include  **skiprows**, **comments**, **delimiter**, **lineterminator**, 
        **uselines**, **usecols**, **excludecols**, **metametadata**, 
        **namesinheader**,**headerlines**, **valuefixer**, **linefixer**, 
        **colfixer**, **delimiter_regex**, **inflines**, **typer**, 
        **missingvalues**, **fillingvalues**, **verbosity**, and various CSV 
        module parameters like **escapechar**, **quoting**, **quotechar**, 
        **doublequote**, **skipinitialspace**.              

    **Returns**

        **R** :  numpy record array

            Record array constructed from data in the SV file

        **metadata** :  dictionary

            Metadata read and constructed during process of reading file.

    **See Also:**

            :func:`tabular.io.loadSVcols`, :func:`tabular.io.saveSV`, 
            :func:`tabular.io.DEFAULT_TYPEINFERER`

    """    
    [columns, metadata] = loadSVcols(fname, **kwargs)
    
    if 'names' in metadata.keys():
        names = metadata['names']
    else:
        names = None
 
    if 'formats' in metadata.keys():
        formats = metadata['formats']
    else:
        formats = None
    
    if 'dtype' in metadata.keys():
        dtype = metadata['dtype']
    else:
        dtype = None
 
    if renamer is not None:
        print 'Trying user-given renamer ...'
        renamed = renamer(names)
        if len(renamed) == len(uniqify(renamed)):
            names = renamed
            print '''... using renamed names (original names will be in return 
                     metadata)'''
        else:
            print '... renamer failed to produce unique names, not using.'
            
    if names and len(names) != len(uniqify(names)):
        print 'Names are not unique, reverting to default naming scheme.'
        names = None


    return [utils.fromarrays(columns, type=np.ndarray, dtype=dtype, 
                             shape=shape, formats=formats, names=names, 
                             titles=titles, aligned=aligned, 
                             byteorder=byteorder), metadata]
    

def loadSVcols(fname, usecols=None, excludecols=None, valuefixer=None, 
               colfixer=None, missingvalues=None, fillingvalues=None,
               typeinferer=None, **kwargs):
    """
    Load a separated value text file to a list of column arrays.

    Basically, this function calls loadSVrecs, and transposes the string-valued 
    row data returned by that function into a Python list of numpy arrays 
    corresponding to columns, each a uniform Python type (int, float, str).  
    Also uses and returns metadata including column names, formats, coloring, 
    &c. if these items  are determined during the loading process.

    **Parameters**

        **fname** :  string or file object

            Path (or file object) corresponding to a separated variable (CSV) 
            text file.
                    
        **usecols** :  sequence of non-negative integers or strings, optional

            Only the columns in *usecols* are loaded and processed.  Columns can 
            be described by number, with 0 being the first column; or if name 
            metadata is present, then by name; or, if color group information is 
            present in the file, then by color group name.  Default is None, 
            e.g. all columns are loaded.
 
        **excludecols** :  sequence of non-negative integers or strings, optional

            Converse of **usecols**, e.g. all columns EXCEPT those listed 
            will be loaded. 
            
        **valuefixer**  :  callable, or list or dictionary of callables, optional
    
            These callable(s) are applied to every value in each field.  The 
            application is done after line strings are loaded and split into 
            fields, but before any typing or missing-value imputation is done.  
            The purpose of the **valuefixer** is to prepare column 
            values for typing and imputation.  The valuefixer callable can 
            return a string or a python object.  If `valuefixer` is a single 
            callable, then that same callable is applied to values in all 
            column; if it is a dictionary, then the keys can be either 
            numbers or names and the value for the key will be applied to 
            values in the corresponding column with that name or number; if 
            it is a list, then the list elements must be in 1-to-1 
            correspondence with the loaded columns, and are applied to each 
            respectively.
               
        **colfixer** : callable, or list or dictionary of callables, optional

            Same as **valuefixer**, but instead of being applied to 
            individual values, are applied to whole columns (and must return 
            columns or numpy arrays of identical length).  Like valuefixer, 
            colfixer callable(s) are applied before typing and missing-value 
            imputation.  
                
        **missingvalues** : string, callable returning string, or list or dictionary of strings or string-valued callable
            
            String value(s) to consider as "missing data" and to be replaced 
            before typing is done.   If specified as a callable, the 
            callable will be applied to the column(s) to determine missing 
            value.  If specified as a dictionary, keys are expected to be 
            numbers of names of columns, and values are individual missing 
            values for those columns (like **valuefixer** inferface).                   
                
        **fillingvalues** : string, pair of strings, callable returning string, or list or dictionary of strings or string-valued callable
            
            Values to be used to replace missing data before typing is done.   
            If specified as a  single non-callable, non-tuple value, this 
            value is used to replace all missing data.  If specified as a 
            callable, the callable is applied to the column and returns the 
            fill value (e.g. to allow the value to depend on the column 
            type).  If specified as a pair of values, the first value acts 
            as the missing value and the second as the value to replace 
            with.  If a dictionary or list of values, then values are 
            applied to corresponding columns.  
    
        NOTE:  all the **missingvalues** and **fillingvalues** 
        functionalities can be replicated (and generalized) using the 
        **valuefixer** or **colfixer** parameters, by specifying function(s) 
        which identify and replace missing values.  While more limited, 
        using **missingvalues** and **fillingvalues**  interface is easier 
        and gives better performance.   
    
        **typer** : callable taking python list of strings (or other values) 
        and returning 1-d numpy array; or list dictionary of such callables  
            
           Function used to infer type and convert string lists into typed 
           numpy arrays, if no format information has been provided.  When 
           applied at all, this function is applied after string have been 
           loaded and split into fields.  This function is expected to 
           impute missing values as well, and will override any setting of 
           **missingvalues** or **fillingvalues**.  If a callable is passed,  
           it is used as typer for all columns, while if a dictionary (or 
           list) of callables is passed, they're used on corresponding 
           columns.  If needed (e.g. because formatting information hasn't 
           been supplied) but **typer** isn't specified (at least, for a 
           given column), the constructor defaults to using the 
           `utils.DEFAULT_TYPEINFERER` function.          
                          
        **kwargs**: keyword argument dictionary of variable length
         
            Contains various parameters to be passed on to loadSVrecs, 
            including **skiprows**, **comments**, **delimiter**, 
            **lineterminator**, **uselines**,  **metametadata**, 
            **namesinheader**,**headerlines**, **linefixer**,  
            **delimiter_regex**, **inflines**, **verbosity**, and various 
            CSV module parameters like **escapechar**, **quoting**, 
            **quotechar**, **doublequote**, **skipinitialspace**. 
            
    **Returns**

        **columns** :  list of numpy arrays

            List of arrays corresponding to columns of data.

        **metadata** :  dictionary

            Metadata read and constructed during process of reading file.

    **See Also:**

            :func:`tabular.io.loadSV`, :func:`tabular.io.saveSV`, 
            :func:`tabular.io.DEFAULT_TYPEINFERER`

    """
    [records, metadata] = loadSVrecs(fname, **kwargs)
     
    lens = np.array([len(r) for r in records])
    assert (lens == lens[0]).all(), 'Not all records have same number of fields'

    l0 = lens[0]
    processmetadata(metadata,items='types,formats', ncols = l0)

    if usecols is not None:
        getcols = [i if i >= 0 else l0 + i for i in usecols 
                   if isinstance(i, int)]
        if 'names' in metadata.keys():
            names = metadata['names']
            getcols += [names.index(c) for c in usecols if c in names]
            if 'coloring' in metadata.keys():
                coloring = metadata['coloring']
                for c in usecols:
                    if c in coloring.keys():
                        getcols += [names.index(n) for n in coloring[c]]
        getcols = uniqify(getcols)
    else:
        if 'names' in metadata.keys():
            names = metadata['names']
            getcols = range(len(names))
        else:
            getcols = range(l0)
        if excludecols is not None:
            dontget = [i if i >= 0 else l0 + i for i in excludecols 
                       if isinstance(i, int)]
            if 'names' in metadata.keys():
                dontget += [names.index(c) for c in excludecols if c in names]
                if 'coloring' in metadata.keys():
                    coloring = metadata['coloring']
                    for c in excludecols:
                        if c in coloring.keys():
                            dontget += [names.index(n) for n in coloring[c]]
            getcols = list(set(getcols).difference(dontget))
    
    getcols.sort()
    if max(getcols) > l0:
        bad = [i for i in getcols if i >= l0]
        getcols = [i for i in getcols if i < l0]
        print 'Too many column names. Discarding columns,', bad
        
    metadatacolthreshold(metadata,getcols)
 
    if 'formats' in metadata.keys() or 'types' in metadata.keys():
        if 'formats' in metadata.keys():
            formats = metadata['formats']
        else:
            formats = metadata['types']
        formats = dict(zip(getcols,formats))
    else:
        formats = dict([(j,None) for j in getcols])
              
    if 'names' in metadata.keys():
        names = metadata['names']
    else:
        names = None
        
    valfix = utils.processvfd(valuefixer, numbers=getcols, names=names)
    colfix = utils.processvfd(colfixer, numbers=getcols, names=names)
    missval = utils.processvfd(missingvalues, numbers=getcols, names=names)
    fillval = utils.processvfd(fillingvalues, numbers=getcols, names=names)
    typer = utils.processvfd(typeinferer, numbers=getcols, names=names)
      
    return [[preparecol(records, j, formats[j], valfix[j], colfix[j],
             missval[j], fillval[j], typer[j]) for j in getcols], metadata]


def preparecol(records, j, format, valfix, colfix, missval, fillval, typer):

    assert (typer is None or (fillval is None or missval is None)), '''If 
           typeinferer is set for a given column then neither fillingvalues or 
           missingvalues can be set for that column'''
     
    if not valfix:
        col = [rec[j] for rec in records]
    else:
        col = [valfix(rec[j]) for rec in records]
    if colfix:
        col = colfix(col)

    if isinstance(fillval, tuple):
        if missval is None and len(fillval) > 1:
            missval = fillval[0]
        fillval = fillval[-1]

    if missval is None:
        missval = utils.DEFAULT_STRINGMISSINGVAL
    else:
        missval = missval(col) if hasattr(missval, '__call__') else missval
        
    if missval in col:  
        is_missing =True
        mcol = np.array(col)
        missing = mcol == missval

        if fillval is None:
            if format:
                fillval = utils.DEFAULT_FILLVAL(np.dtype(format).descr[0][1])
            else:
                fillval = utils.DEFAULT_STRINGMISSINGVAL
        else:
            fillval = fillval(col) if hasattr(fillval,'__call__') else fillval

        for i in missing.nonzero()[0]:
            col[i] = fillval
            

    if format:
        col = utils.makearray(col, format)
    else:
        if not typer:
           typer = utils.DEFAULT_TYPEINFERER
        col = typer(col)
                        
    return col
    

def loadSVrecs(fname, uselines=None, skiprows=0, linefixer=None, 
               delimiter_regex=None, verbosity=DEFAULT_VERBOSITY, **metadata):
    """
    Load a separated value text file to a list of lists of strings of records.

    Takes a tabular text file with a specified delimeter and end-of-line 
    character, and return data as a list of lists of strings corresponding to 
    records (rows).  Also uses and returns metadata (including column names, 
    formats, coloring, &c.) if these items are determined during the loading 
    process.   

    **Parameters**

        **fname** :  string or file object

            Path (or file object) corresponding to a separated variable
            (CSV) text file.
 
        **delimiter** : single-character string
        
            When reading text file, character to use as delimiter to split 
            fields.  If not specified, the delimiter is determined first by 
            looking for special-format metadata specifying the delimiter, and 
            then if no specification is found, attempts are made to infer 
            delimiter from file contents.  (See **inflines** parameter below.)  
            
        **delimiter_regex** : regular expression (compiled or in string format)                    
         
            Regular expression to use to recognize delimiters, in place of a 
            single character.  (For instance, to have whitespace delimiting, 
            using delimiter_regex = '[\s*]+')
                         
        **lineterminator** : single-character string
        
            Line terminator to use when reading in using SVfile.
            
        **skipinitialspace** : boolean
        
            If true, strips whitespace following the delimiter from field.   
            
       The **delimiter**, **linterminator** and **skipinitialspace** 
       parameters are passed on as parameters to the python CSV module, which is 
       used for reading in delimited text files.  Additional parameters from 
       that interface that are replicated in this constructor include 
       **quotechar**, **escapechar**, **quoting**, **doublequote** and 
       **dialect** (see CSV module documentation for more information).

        **skiprows** :  non-negative integer, optional

            When reading from a text file, the first `skiprows` lines are 
            ignored.  Default is 0, e.g no rows are skipped. 

        **uselines** : pair of non-negative integer, optional
        
            When reading from a text file, range of lines of data to load.  (In 
            contrast to **skiprows**, which specifies file rows to ignore 
            before looking for header information, **uselines** specifies which 
            data (non-header) lines to use, after header has been striped and 
            processed.)  See **headerlines** below.
            
        **comments** : single-character string, optional
            
            When reading from a text file, character used to distinguish header 
            lines.  If specified, any lines beginning with this character at the 
            top of the file are assumed to contain header information and not 
            row data. 
      
        **headerlines** : integer, optional

            When reading from a text file, the number of lines at the top of the 
            file (after the first  `skiprows` lines) corresponding to the header 
            of the file, where metadata can be found.  Lines after headerlines 
            are assumed to contain row contents.  If not specified, value is 
            determined first by looking for special metametadata  in first line 
            of file (see Tabular reference documentation for more information 
            about this), and if no such metadata is found, is inferred by 
            looking at file contents.    
            
        **namesinheader** : Boolean, optional

            When reading from a text file, if `namesinheader == True`, then 
            assume the column names are in the last header line (unless 
            overridden by existing metadata or metametadata directive).  Default 
            is True.                        
            
        **linefixer** : callable, optional

           This callable is applied to every line in the file.  If specified, 
           the called is applied directly to the strings in the file, after 
           they're split in lines but before they're split into fields.  The 
           purpose is to make lines with errors or mistakes amenable to 
           delimiter inference and field-splitting. 
            
        **inflines** :  integer, optional
        
            Number of lines of file to use as sample data when inferring 
            delimiter and header.   

        **metametadata** :  dictionary of integers or pairs of integers
            
            Specifies supplementary metametadata information for use 
            with SVfile loading.  See Tabular reference documentation for more 
            information
            
    **Returns**

            **records** :  list of lists of strings

                List of lists corresponding to records (rows) of data.

            **metadata** :  dictionary

                Metadata read and constructed during process of reading file.

    **See Also:**

            :func:`tabular.io.loadSV`, :func:`tabular.io.saveSV`, 
            :func:`tabular.io.DEFAULT_TYPEINFERER`

    """
    if delimiter_regex and isinstance(delimiter_regex, types.StringType):
        import re
        delimiter_regex = re.compile(delimiter_regex) 
   
    [metadata, inferedlines, WHOLETHING] = getmetadata(fname, skiprows=skiprows,
                                                linefixer=linefixer, 
                                                delimiter_regex=delimiter_regex, 
                                                verbosity=verbosity, **metadata)

    if uselines is None:
        uselines = (0,False)
    
    if is_string_like(fname):
        fh = file(fname, 'rU')
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle') 
 
    for _ind in range(skiprows+uselines[0] + metadata['headerlines']):
        fh.readline()
        
    if linefixer or delimiter_regex:
        fh2 = tempfile.TemporaryFile('w+b')
        F = fh.read().strip('\n').split('\n')
        if linefixer:
            F = map(linefixer,F)
        if delimiter_regex:
            F = map(lambda line: 
                    delimiter_regex.sub(metadata['dialect'].delimiter, line), F)       
        fh2.write('\n'.join(F))        
        fh2.seek(0)
        fh = fh2        

    reader = csv.reader(fh, dialect=metadata['dialect'])

    if uselines[1]:
        linelist = []
        for ln in reader:
            if reader.line_num <= uselines[1] - uselines[0]:
                linelist.append(ln)
            else:
                break
    else:
        linelist = list(reader)
      
    fh.close()

    if linelist[-1] == []:
        linelist.pop(-1)

    return [linelist,metadata]      


def getmetadata(fname, inflines=2500, linefixer=None, delimiter_regex=None, 
                namesinheader=True, skiprows=0, verbosity=DEFAULT_VERBOSITY, 
                comments=None, **metadata):

    metadata = dict([(k,v) for (k,v) in metadata.items() if v is not None]) 

    if comments is None:
        comments = '#'

    if 'metametadata' in metadata.keys():
        mmd = metadata['metametadata']
    else:
        mmd = None
   
    if 'formats' in metadata.keys() and is_string_like(metadata['formats']):
        metadata['formats'] = metadata['formats'].split(',')
        
    storedmetadata = readstoredmetadata(fname, skiprows=skiprows, 
                                        comments=comments, metametadata=mmd, 
                                        verbosity=verbosity)
    
    if storedmetadata:
        if verbosity > 7:
            print '\n\nStored metadata read from file:', storedmetadata, '\n\n' 
        for name in storedmetadata:
            if ((name in metadata.keys()) and (storedmetadata[name] != None) and 
                (storedmetadata[name] != metadata[name])):
                if verbosity >= 4:
                    print '''WARNING:  A value for %s was found in metadata 
                             read from special-format header in file %s as well 
                             as being provided explicitly, and read value 
                             differs from provided value.  Using provided 
                             value.''' % (name, fname)
            else:
                metadata[name] = storedmetadata[name]

    if is_string_like(fname):
        fh = file(fname, 'rU')
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
        
    for _ind in range(skiprows):
        fh.readline()
  
   
    WHOLETHING = False
    if inflines is None:
        F = fh.read().strip('\n').split('\n')
    else:
        if 'headerlines' in metadata.keys():
            inflines += metadata['headerlines']
        F = [fh.readline().strip('\n') for i in range(inflines)]
        if len(F) > 0 and F[-1] == '':
            WHOLETHING = True
        F = [f for f in F if f != '']
    
    fh.close()      
    
    if linefixer:
        F = map(linefixer, F)
    
    if 'headerlines' not in metadata.keys():
       metadata['headerlines'] = inferheader(F, metadata=metadata, 
                                             verbosity=verbosity)
       
    if namesinheader and ((metadata['headerlines'] == 0) or 
                          (metadata['headerlines'] is None)):
            metadata['headerlines'] = 1
            if verbosity >= 6:
                print '''... assuming "headerlines" = 1, since 
                         "namesinheader" = True.'''  
    
    delimiter_infer = False
    if 'dialect' not in metadata.keys():
        if 'delimiter' not in metadata.keys():
            delimiter_infer = True
            infdia = inferdialect(fname=fname, 
                                  datalines=F[metadata['headerlines']:],
                                  delimiter_regex=delimiter_regex)
            metadata['dialect'] = infdia
            printdialectinferencemessage(metadata['dialect'], verbosity, 
                                         delimiter_regex)
        else:
            metadata['dialect'] = csv.Sniffer().sniff(metadata['delimiter'])
 
    processmetadata(metadata, items='dialect', comments=comments, 
                    verbosity=verbosity)

    if delimiter_infer:
        printdelimitercheckmessage(metadata['dialect'], infdia, storedmetadata,
                                   verbosity)

    if 'names' not in metadata.keys() and namesinheader and not storedmetadata:
        assert metadata['headerlines'] > 0, '''Trying to set names using last 
        header line since namesinheader is True, but "headerlines" = 0 
        indicating no headerline present at all.'''
        metadata['names'] = F[metadata['headerlines'] - 1]
        if verbosity > 1:
            print 'Inferring names from the last header line (line', metadata['headerlines'], ').'

    processmetadata(metadata, items='names,formats,types', comments=comments, 
                    delimiter_regex=delimiter_regex, verbosity=verbosity)

    return [metadata, F, WHOLETHING]
    
 
def printdialectinferencemessage(dialect, verbosity, delimiter_regex):
    if 8 > verbosity > 2:  
        if delimiter_regex:
            print 'Using delimiter_regex with representative delimiter ' + repr(dialect.delimiter)
        else:
            print 'Inferring delimiter to be ' + repr(dialect.delimiter)
    elif verbosity >= 8:
        print 'Inferring dialect with values:', printdialect(dialect)


def printdelimitercheckmessage(dialect,infdia,storedmetadata,verbosity):
    if infdia.delimiter != dialect.delimiter:
        if verbosity >= 5:
            if (storedmetadata and ('delimiter' in storedmetadata.keys()) and 
                (infdia.delimiter == storedmetadata['delimiter'])):
                print '''Inferred delimiter differs from given delimiter but 
                         equals delimiter read from metadata in file.  Are you 
                         sure you haven\'t made a mistake?'''
            else:
                print 'Inferred delimiter differs from given delimiter.'


def saveSV(fname, X, comments=None, metadata=None, printmetadict=None,
                   dialect=None, delimiter=None, doublequote=True, 
                   lineterminator='\n', escapechar = None, 
                   quoting=csv.QUOTE_MINIMAL, quotechar='"', 
                   skipinitialspace=False, stringifier=None,
                   verbosity=DEFAULT_VERBOSITY):
    """
    Save a tabarray to a separated-variable (CSV) file.

    **Parameters**

        **fname** :  string

            Path to a separated variable (CSV) text file.

        **X** :  tabarray

            The actual data in a :class:`tabular.tab.tabarray`.

        **comments** :  string, optional

            The character to be used to denote the start of a header (non-data) 
            line, e.g. '#'.  If not specified, it is determined according to the 
            following rule:  '#' if `metadata` argument is set, otherwise ''.

        **delimiter** :  string, optional

            The character to beused to separate values in each line of text, 
            e.g. ','.  If not specified, by default, this is inferred from 
            the file extension: if the file ends in `.csv`, the delimiter is 
            ',', otherwise it is '\\t.'

        **linebreak** :  string, optional

            The string separating lines of text.  By default, this is assumed to 
            be '\\n', and can also be set to be '\\r' or '\\r\\n'.

        **metadata** :  list of strings or Boolean, optional

            Allowed values are True, False, or any sublists of the list 
            `['names', 'formats', 'types', 'coloring', 'dialect']`.  These 
            keys indicate what special metadata is printed in the header.

            * If a sublist of 
             `['names', 'formats', 'types', 'coloring', 'dialect']`, then the 
             indicated types of metadata are written out.  

            * If `True`, this is the same as 
              `metadata = ['coloring', 'types', 'names','dialect']`, e.g. as 
               many types of metadata as this algorithm currently knows how to 
               write out. 

            * If 'False', no metadata is printed at all, e.g. just the data.
                    
            * If `metadata` is not specified, the default is `['names']`, that 
              is, just column names are written out.
                        
        **printmetadict** :  Boolean, optional

            Whether or not to print a string representation of the 
            `metadatadict` in the first line of the header.

            If `printmetadict` is not specified, then:

            * If `metadata` is specified and is not `False`, then
              `printmetadata` defaults to `True`.

            * Else if `metadata` is `False`, then `printmetadata` defaults 
              to `False`.

            * Else `metadata` is not specified, and `printmetadata` defaults 
              to `False`.

            See the :func:`tabular.io.loadSV` for more information about 
            `metadatadict`.
            
        **stringifier** : callable 
        
            Callable taking 1-d numpy array and returning Python list of strings 
            of same length, or dictionary or tuple of such callables.  
                    
            If specified, the callable will be applied to each column, and the 
            resulting list of strings will be written to the file.  If 
            specified as a list or dictionary of callables, the functions will 
            be applied to correponding columns.  The default used if 
            **stringifier** is not specified, is `tb.utils.DEFAULT_STRINGIFIER`, 
            which merely passes through string-type columns, and converts 
            numerical-type columns directly to corresponding strings with NaNs 
            replaced with blank values.  The main purpose of specifying a 
            non-default value is to encode numerical values in various string 
            encodings that might be used required for other applications like 
            databases.                    
                               
            NOTE:  In certain special circumstances (e.g. when the 
            lineterminator or delimiter character appears in a field of the 
            data), the Python CSV writer is used to write out data.  To allow 
            for control of the operation of the writer in these circumstances, 
            the following other parameters replicating the interface of the CSV 
            module are also valid, and values will be passed through:  
            **doublequote**, **escapechar**, **quoting**, **quotechar**, and 
            **skipinitialspace**.  (See Python CSV module documentation for more 
            information.)     

    **See Also:**

            :func:`tabular.io.loadSV`

    """    
    if metadata is None:
        metakeys = ['names']
        if printmetadict is None:
            printmetadict = False
            if verbosity > 8:
                print '''Defaulting to not printing out the metametadata 
                         dictionary line.'''
        if comments is None:
            comments = ''
            if verbosity > 8:
                print 'Defaulting empty comment string.'
        if verbosity > 7:
            print 'Defaulting to writing out names metadata.'
    elif metadata is True:
        
        metakeys = defaultmetadatakeys(X)
        
        if printmetadict is None:
            printmetadict = True
            if verbosity > 8:
                print '''Defaulting to printing out the metametadata dictionary 
                         line.'''
        if comments is None:
            comments = ''
            if verbosity > 8:
                print 'Defaulting empty comment string.'            
        if verbosity >= 5:
            print 'Writing out all present metadata keys ... '
    elif metadata is False:
        metakeys = []
        printmetadict = False
        comments = ''
        if verbosity >= 5:
            print 'Writing out no metadata at all.'
    else:
        metakeys = metadata
        if printmetadict is None:
            if metakeys == []:
                printmetadict = False
            else:
                printmetadict = True
        if comments is None:
            comments = ''
        if verbosity >= 5:
            print '''Using user-specified metadata keys to contol metadata 
                     writing.'''
            
    assert lineterminator in ['\r','\n','\r\n'], '''lineterminator must be one 
                                              of ''' + repr( ['\r','\n','\r\n'])
    dialect = getdialect(fname, dialect, delimiter, lineterminator, doublequote, 
                         escapechar, quoting, quotechar, skipinitialspace)
    delimiter = dialect.delimiter     
    
    if 6 > verbosity > 2:
        print 'Using delimiter ', repr(delimiter)
    elif verbosity >= 6:
        print 'Using dialect with values:', repr(printdialect(dialect))
            
    metadata = getstringmetadata(X,metakeys,dialect)
    
    metametadata = {}
    v = 1
    for k in metakeys:
        if k in metadata.keys():
            nl = len(metadata[k].split(lineterminator))
            metametadata[k] = v if nl == 1 else (v, v + nl)
            v = v + nl

    F = open(fname,'wb')

    if printmetadict is True:
        line = "metametadata=" + repr(metametadata)
        F.write(comments + line + lineterminator)

    for k in metakeys:
        if k in metadata.keys():
            for line in metadata[k].split(lineterminator):
                F.write(comments + line + lineterminator)
        
    Write(X, F, dialect, stringifier=stringifier)
    
    F.close()

def defaultmetadatakeys(X):
    if hasattr(X,'metadata') and hasattr(X.metadata,'keys'):
        dk = X.metadata.keys()
        badlist = ['headerlines','metametadata','dialect','coloring', 'types', 'names']
        for b in badlist:
             if b in dk:
                dk.remove(b)
    else:
        dk = []
    dk +=  ['dialect', 'coloring', 'types', 'names']
    dk = uniqify(dk)
    return dk

    
def Write(X, F, dialect, order=None, stringifier=None):

    delimiter = dialect.delimiter
    lineterminator = dialect.lineterminator
    
    if order is None:
        Order = X.dtype.names
    else:
        Order = [x for x in order if x in X.dtype.names]
        assert len(Order) > 0
        
    stringifier = utils.processvfd(stringifier, names=Order)    
  
    ColStr = []
    UseComplex = False
    for (i,name) in enumerate(Order):
        typename = X.dtype[name].name
        D = X[name]
        
        if D.ndim > 1:
            D = D.flatten()
        
        if stringifier[i] == None:
            stringifier[i] = utils.DEFAULT_STRINGIFIER
        ColStr.append(stringifier[i](D))
        
        if typename.startswith('str'):
            if any([delimiter in d or lineterminator in d for d in D]) > 0:
                print("WARNING: An entry in the '" + name +
                      "' column contains at least one instance of the "
                      "delimiter '" + delimiter + "' and therefore will use "
                      "the Python csv module quoting convention (see online " 
                      "documentation for Python's csv module).  You may want "
                      "to choose another delimiter not appearing in records, " 
                      "for performance reasons.")
                UseComplex = True
            elif any([lineterminator in d for d in D]):
                print("WARNING: An entry in the '" + name +
                      "' column contains at least one instance of the "
                      "line terminator '" + lineterminator + "' and therefore "
                      "will use the Python csv module quoting convention (see "
                      "online documentation for Python's csv module).  You may "
                      "want to choose another delimiter not appearing in "
                      "records, for performance reasons.") 
                UseComplex = True

    if UseComplex is True:
        csv.writer(F, dialect=dialect).writerows(([col[i] for col in ColStr] 
                                                for i in range(len(ColStr[0]))))
    else:
        F.write(lineterminator.join([delimiter.join([col[i] for col in ColStr]) 
                              for i in range(len(ColStr[0]))]) + lineterminator)


def printdialect(d):
    return dict([(a,getattr(d, a)) for a in dir(d) if not a.startswith('_')])


def metadatacolthreshold(metadata, getcols):
    getcols = getcols[:]
    getcols.sort()
    if 'names' in metadata.keys():
        n = metadata['names'][:]
        metadata['names'] = [n[i] for i in getcols]
        if 'coloring' in metadata.keys():
            coloring = metadata['coloring']
            metadata['coloring'] = thresholdcoloring(coloring, metadata['names'])
    if 'formats' in metadata.keys():
        f = metadata['formats'][:]
        metadata['formats'] = [f[i] for i in getcols]
    if 'types' in metadata.keys():
        f = metadata['types'][:]
        metadata['types'] = [f[i] for i in getcols]


def inferdialect(fname=None, datalines=None, delimiter_regex=None, 
                 verbosity=DEFAULT_VERBOSITY):
    """
    Attempts to convert infer dialect from csv file lines. 
    
    Essentially a small extension of the "sniff" function from Python CSV 
    module.   csv.Sniffer().sniff attempts to infer the delimiter from a 
    putative delimited text file by analyzing character frequencies.  This 
    function adds additional analysis in which guesses are checked again the 
    number of entries in each line that would result from splitting relative to 
    that guess. If no plausable guess if found, delimiter is inferred from file 
    name  ('csv' yields ',', everything else yields '\t'.)
    
    **Parameters** 
    
        **fname** : pathstring
        
            Name of file.
        
        **datalines** : list of strings
        
            List of lines in the data file.
            
        **lineterminator** : single-character string
        
            Line terminator to join split/join line strings.
        
    **Returns**
    
        csv.Dialect obejct      
    
    """
    if datalines is None:
        if is_string_like(fname):
            fh = file(fname, 'rU')
        elif hasattr(fname, 'readline'):
            fh = fname
        else:
            raise ValueError('fname must be a string or file handle')

        datalines = fh.read().strip().split('\n')
        fh.close() 
    
    if delimiter_regex:
        matches = []
        for l in datalines[:10]:
            matches += delimiter_regex.findall(l)
        poss = {}
        for m in matches:
            for x in set(m):
                poss[x] = m.count(x) + (poss[x] if x in poss.keys() else 0) 
        MaxVal = max(poss.values())
        assert MaxVal > 0, 'delimiter_regex found no matches'
        amax = [x for x in poss.keys() if poss[x] == MaxVal][0]
        return csv.Sniffer().sniff(amax)

    else:
        if not is_string_like(fname):
            fname = None
    
        tries = [10, 30, 60, 100, 200, 400, 800]
    
        if len(datalines) > 100:
            starts = [int(len(datalines) / 5) * i for i in range(5)]
        else:
            starts = [0, int(len(datalines) / 2)]
            
        G = []
        for s in starts:
            for t in [tt for (i, tt) in enumerate(tries) 
                      if i == 0 or s + tries[i-1] <= len(datalines)]:
                try:
                    g = csv.Sniffer().sniff('\n'.join(datalines[s:(s+t)]))
                except:
                    pass
                else:
                    G += [g]
                    break
    
                    
        delims = [g.delimiter for g in G]   
        G = [g for (i, g) in enumerate(G) if g.delimiter not in delims[:i]]
        V = []
        for g in G:
            lvec = np.array([len(r) for r in 
                             list(csv.reader(datalines[:1000], dialect=g))])  
            V += [lvec.var()]
    
    
        if len(G) > 0:
            V = np.array(V)
            if V.min() > 0:
                fnamedelim = inferdelimiterfromname(fname)
                if fnamedelim not in delims:
                    fnamevar = np.array([len(r) for r in 
                                         list(csv.reader(datalines[:1000], 
                                                  delimiter=fnamedelim))]).var()
                    if fnamevar < V.min():
                        return csv.Sniffer().sniff(fnamedelim)
            return G[V.argmin()]
        else:
            if verbosity > 2:
                print 'dialect inference failed, infering dialect to be', inferdelimiterfromname(fname) , 'from filename extension.'
            return csv.Sniffer().sniff(inferdelimiterfromname(fname))



def readstoredmetadata(fname, skiprows=0, linenumber=None, comments='#', 
                       metametadata=None, verbosity=DEFAULT_VERBOSITY):
    """
    Read metadata from a delimited text file.
    
    """
    if is_string_like(fname):
        fh = file(fname, 'rU')
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
       
    if not metametadata:
        for _ind in range(skiprows):
            fh.readline()

        phlines = []
        if linenumber is None:
            if comments:
                for line in fh:
                    if not line.startswith(comments):
                        if len(phlines) == 0:
                            phlines = [line]
                        break
                    else:
                        phlines.append(line)
                if len(phlines) == 1:
                    if verbosity >= 10:
                        print '''Looking for metametadata on line 0 
                                 (no comment lines present).'''
                else:
                    if verbosity >= 9:
                        print '''Searching for metametadata lines up to 
                                 and including line %d where comments 
                                 end.''' % (len(phlines) - 1)
            else:
                phlines = [fh.readline()]
                if verbosity >=9:
                    print '''No comments found, looking for metametadata on 
                             line 0.'''
        else:
            for _ind in range(linenumber):
                fh.readline()
            phlines = [fh.readline()]
                               
        metametadata = None
        for (ln, metametaline) in enumerate(phlines):
            s = re.compile(r'metametadata[\s]*=[\s]*{').search(metametaline)
            if s:
                l = s.start()
                if len(uniqify(metametaline[:l])) <= 1:
                    metametaline = metametaline[l:].rstrip()
                    try:
                        X = compiler.parse(metametaline)
                    except SyntaxError:
                        pass
                    else:
                        if IsMetaMetaDict(X):
                            exec metametaline
                            if verbosity > 6:
                                print 'Found valid metametadata at line', ln, 'in file.  Metametadata is:', metametadata
                            break
   
    if metametadata:
        
        metadata = {}
        metadata['metametadata'] = metametadata
        Tval = max([v if isinstance(v,int) else max(v) 
                    for v in metametadata.values()])
        fh = file(fname, 'rU')
        data = [fh.readline() for _ind in range(Tval + 1 + skiprows)][skiprows:]
 
        if (max([v if isinstance(v,int) else max(v) 
                for v in  metametadata.values()]) < len(data)):
            for n in metametadata.keys():
                [s, e] = [metametadata[n], metametadata[n]+1] \
                          if isinstance(metametadata[n],int) \
                          else [metametadata[n][0],metametadata[n][1]]
                metadata[n] = ''.join(data[s:e]).strip('\n')
            processmetadata(metadata, comments=comments, verbosity=verbosity)
        
            return metadata


def processmetadata(metadata, items=None, comments=None, delimiter_regex=None,
                    ncols=None, verbosity=DEFAULT_VERBOSITY):
    """
    Process Metadata from stored (or "packed") state to functional state.

    Metadata can come be read from a file "packed" in various ways, 
    e.g. with a string representation of a dialect or coloring dictionary.  
    This function "unpacks" the stored metadata into useable Python
    objects.  It consists of a list of quasi-modular parts, one for each 
    type of recognized metadata.

    **Parameters**

        **metadata** : dictionary

            This argument is a dictionary whose keys are strings denoting
            different kinds of metadata (e.g. "names" or "formats") and whose 
            values are the metadata of that type.  The metadata dictionary is 
            modified IN-PLACE by this function.

        **items** : string or list of strings, optional

            The items arguments specifies which metadata keys are to be 
            processed.  E.g. of items = 'names,formats', then the "names" 
            metadata and "formats" metadata will be processed, but no others.
            Note however, that sometimes, the processing of one type of metadata 
            requires that another be processed first, e.g. "dialect" must 
            processed into an actual CSV.dialect object before "names"  is 
            processed.  (The processed of "names" metadata involves splitting 
            the names metadata string into a list, using the delimiter.  This 
            delimiter is part of the dialect object.)   In these cases, if you 
            call processmetadata on one item before its requirements are 
            processed, nothing will happen.

        **comments** : single-character string, optional

            The comments character is used to process many pieces of metadata, 
            e.g. it is striped of the left side of names and formats strings
            before splitting on delimiter.

        **verbosity** : integer, optional

            Determines the level of verbosity in the printout of messages
            during the running of the procedure. 

   **Returns**
   
       Nothing.

    """
    items = items.split(',') if isinstance(items,str) else items

    if comments is None:
        if 'comments' in metadata.keys():
            comments = metadata['comments']
        else:
            comments = '#'
            if verbosity > 8:
                print 'processing metadata with comments char = #'
    else:
        if (('comments' in metadata.keys()) and 
            (comments != metadata['comments']) and (verbosity > 8)):
            print 'comments character specified to process metadata (', repr(comments) ,') is different from comments charcater set in metadata dictionary (', repr(metadata['comments']) , ').'
    
    if not items:
        for k in metadata.keys():
            if is_string_like(metadata[k]):
                metadata[k] = '\n'.join([x.lstrip(comments) 
                                         for x in metadata[k].split('\n') ])
    
    if not items or 'dialect' in items:
        if 'dialect' in metadata.keys():
            if isinstance(metadata['dialect'],str):
                D = dialectfromstring(metadata['dialect'].lstrip(comments))
                if D:
                    metadata['dialect'] = D
                    if (verbosity > 8):
                        print 'processed dialect from string'
                        
                else:
                    if (verbosity > 8):
                        print '''Dialect failed to be converted properly from 
                                 string representation in metadata.'''

            if 'delimiter' in dir(metadata['dialect']):
                for a in dir(metadata['dialect']):
                    if not a.startswith('_') and a in metadata.keys():
                        setattr(metadata['dialect'],a, metadata[a])
                        if ((verbosity > 2 and a == 'delimiter') or 
                            (verbosity >= 8)):
                            print 'Setting dialect attribute', a, 'to equal specified value:', repr(metadata[a])
                    elif not a.startswith('_') and a not in metadata.keys():
                        metadata[a] = getattr(metadata['dialect'], a)
                        if ((verbosity > 2 and a == 'delimiter') or (verbosity >= 8)):
                            print 'Setting metadata attribute from dialect', a , 'to equal specified value:', repr(metadata[a])

    if (not items or 'names' in items) and ('names' in metadata.keys()): 
        if is_string_like(metadata['names']):

            if delimiter_regex:
                metadata['names'] = delimiter_regex.split(metadata['names'])
            elif (('dialect' in metadata.keys()) and 
                  ('delimiter' in dir(metadata['dialect']))):
                d = metadata['dialect']
                n = metadata['names']
                metadata['names'] = list(csv.reader([n.lstrip(comments)],      
                                                     dialect=d))[0]
                if (verbosity > 8):
                    print '... splitting "names" metadata from string with delimiter', repr(d.delimiter), '. Resulting names:', metadata['names']

    if (not items or 'formats' in items) and 'formats' in metadata.keys(): 
        if is_string_like(metadata['formats']):   
            if delimiter_regex:
                metadata['formats'] = delimiter_regex.split(metadata['formats']) 
            elif (('dialect' in metadata.keys()) and ('delimiter' in dir(metadata['dialect']))):
                d = metadata['dialect']
                n = metadata['formats']
                metadata['formats'] = list(csv.reader([n.lstrip(comments)],  
                                           dialect=d))[0]

                if (verbosity > 8):
                    print '... splitting "formats" metadata from string with delimiter', repr(d.delimiter), '. Resulting names:', metadata['formats']       
 
        if ncols:
            metadata['formats'] = postprocessformats(metadata['formats'],ncols)


    if (not items or 'types' in items) and 'types' in metadata.keys(): 
        if is_string_like(metadata['types']):      
            if delimiter_regex:
                metadata['types'] = delimiter_regex.split(metadata['types']) 
            elif (('dialect' in metadata.keys()) and 
                  ('delimiter' in dir(metadata['dialect']))):
                d = metadata['dialect']
                n = metadata['types']
                metadata['types'] = list(csv.reader([n.lstrip(comments)],  
                                         dialect=d))[0]
                if (verbosity > 8):
                    print '... splitting "types" metadata from string with delimiter', repr(d.delimiter), '. Resulting names:', metadata['types']      
        
        if ncols:
            metadata['types'] = postprocessformats(metadata['types'],ncols)

    if (not items or 'coloring' in items) and ('coloring' in metadata.keys()):
        if is_string_like(metadata['coloring']):
            C = coloringfromstring(metadata['coloring'].lstrip(comments))
            if C:
                metadata['coloring'] = C
                if (verbosity > 8):  
                    print '... processed coloring from string'
            else:
                if verbosity > 1:
                    print 'Coloring failed to be converted properly from string representation in metadata ; removing coloring data from active metadata (putting it in "loaded_coloring").'
                    metadata['loaded_coloring'] = metadata['coloring']
                    metadata.pop('coloring')

    if (not items or 'headerlines' in items):
        if 'headerlines' in metadata.keys():
            if isinstance(metadata['headerlines'], str):
                try:
                    h = metadata['headerlines']
                    metadata['headerlines'] = int(h.lstrip(comments))
                except (ValueError,TypeError):
                    if verbosity > 6:
                        print 'headerlines metadata failed to convert to an integer.'
                else:
                    pass
                        

            if isinstance(metadata['headerlines'], int):
                if 'metametadata' in metadata.keys():
                    h= metadata['headerlines']
                    mmd = metadata['metametadata']
                    metadata['headerlines'] = max(h, 1 + max([v 
                                               if isinstance(v, int) else max(v) 
                                               for v in mmd.values()]))
                    if ((metadata['headerlines'] > h) and (verbosity > 8)):
                        print 'Resetting headerlines from', h, 'to', metadata['headerlines'], 'because of line number indications from metametadata.'


def postprocessformats(formats, ncols):
    if (isinstance(formats, list) and (len(formats) == 1)):
        return formats * ncols
    else:
        return formats


def inferheader(lines, comments=None, metadata=None,
                verbosity=DEFAULT_VERBOSITY):
    """
    Infers header from a CSV or other tab-delimited file.
    
    This is essentially small extension of the csv.Sniffer.has_header algorithm.
    provided in the Python csv module.   First, it checks to see whether a 
    metametadata dictionary is present, specifiying the lines numbers of 
    metadata lines in the header, and if so, sets the header lines to include
    at least those lines.  Then iookms to see if a comments character is 
    present, and if so, includes those lines as well.  If either of the above 
    returns a nono-zero number of headerlines, the function returns that 
    number; otherwise, it uses the csv.Sniffer module, checking each line in 
    succession, and stopping at the first line where the sniffer module finds no 
    evidence of a header, and returning that line numner.
    
    **Parameters** 
    
        **lines** : line of strings 
        
            The list of lines representing lines in the file
            
        **comments** : single-character string, optional
        
            Comments character  specification. 
            
        **metadata** : metadata dictionary, optional
        
            Used to determine a comments character and metametadata dicationary, 
            if present.

    **Returns**
    
        Integer, representing the number of (inferred) header lines at the top 
        of the file.
    
    """
     
    if ((comments is None) and metadata and ('comments' in metadata.keys())):
        comments = metadata['comments']
    if (comments is None):
        comments = '#'  

    if ('metametadata' in metadata.keys()):
        mmd = metadata['metametadata']
        cc = 1 + max([v if isinstance(v, int) else max(v) 
                      for v in mmd.values()])
    else:
        cc = 0
        
    if (comments != ''):
        if (cc < len(lines)):
            for l in xrange(cc,len(lines)):
                if not lines[l].startswith(comments):
                    break
        else:
            l = cc

    if (l > 0):
        return l
    else:
        for j in xrange(min(1000, len(lines))):
            hasheader = 'unset'
            for k in [100, 200, 400, 800, 1600]:
                F = '\n'.join(lines[j:(j+k)])
                try:
                    hasheader = csv.Sniffer().has_header(F)
                except:
                    pass
                else:
                    break
            if not hasheader:
                return j
           

def isctype(x, t):
    a = (isinstance(x, Const) and isinstance(x.value, t))
    b = t == types.BooleanType and isinstance(x, Name) and \
                     x.name in ['False', 'True']
    c =  (t == types.NoneType and isinstance(x, Name) and x.name == 'None')
    return a or b or c


def IsMetaMetaDict(AST):
    """
    Checks whether a given AST (abstract syntax tree) object represents a 
    metametadata dictionary.   
    
    """
    isintpair = lambda x : (isinstance(x,Tuple) and (len(x.asList()) == 2) and 
                            isctype(x.asList()[0], int) and 
                            isctype(x.asList()[1], int))
    try:
        if ((len(AST.getChildren()) > 1) and 
            isinstance(AST.getChildren()[1], Stmt)):
            if isinstance(AST.getChildren()[1].getChildren()[0], Assign):
                [s, d] = AST.getChildren()[1].getChildren()[0].asList()
    except (TypeError, AttributeError):
        return False
    else:
        if (isinstance(s, AssName) and s.name == 'metametadata'):     
            if isinstance(d, Dict):
                return all([isctype(k, str) and 
                            (isctype(v, int) or isintpair(v)) 
                            for (k,v) in d.items])

def dialectfromstring(s):
    """
    Attempts to convert a string representation of a CSV 
    dialect (as would be read from a file header, for instance) 
    into an actual csv.Dialect object.   
    
    """
    try:
        AST = compiler.parse(s)
    except SyntaxError:
        return
    else:
        try:
            if (len(AST.getChildren()) > 1):
                ST = AST.getChildren()[1]
                if isinstance(ST, Stmt):
                    if isinstance(ST.getChildren()[0], Discard):
                        d = ST.getChildren()[0].asList()[0]
        except (TypeError,AttributeError):
            pass
        else:
            if (isinstance(d,Dict) and (len(d.items) > 0)):
                if all([isctype(i[0], str) for i in d.items]):
                    testd = csv.Sniffer().sniff('a,b,c')
                    if all([n.value in dir(testd) and 
                        isctype(v, type(getattr(testd, n.value))) for (n,v) in 
                                                                      d.items]):
                        D = eval(s)
                        for n in D.keys():
                            setattr(testd, n, D[n])
                        return testd


def coloringfromstring(s):
    """
    Attempts to convert a string representation of a coloring dictionary (as 
    would be read from a file header, for instance) into an actual coloring 
    dictionary.
    
    """
    try:
        AST = compiler.parse(s)
    except SyntaxError:
        return
    else:
        try:
            if len(AST.getChildren()) > 1:
                ST = AST.getChildren()[1]
                if isinstance(ST, Stmt) and isinstance(ST.getChildren()[0],
                                                                     Discard):
                    d = ST.getChildren()[0].asList()[0]
        except (TypeError, AttributeError):
            pass
        else:
            if isinstance(d,Dict) and len(d.items) > 0:
                if all([isctype(i[0],str) for i in d.items]):
                    if all([isinstance(i[1],List) for i in d.items]):
                        if all([all([isctype(j,str) for j in i[1]]) for i in 
                                                                      d.items]):
                            return eval(s)


def getdialect(fname, dialect, delimiter, lineterminator, doublequote,
               escapechar, quoting, quotechar, skipinitialspace):
                                               
    if dialect is None:
        if delimiter is None:
            dialect = csv.Sniffer().sniff(inferdelimiterfromname(fname))
        else:
            dialect = csv.Sniffer().sniff(delimiter)
           
    dialect.lineterminator = lineterminator
    if doublequote is not None:
        dialect.doublequote = doublequote
    if escapechar is not None:
        dialect.escapechar = escapechar
    if quoting is not None:
        dialect.quoting = quoting
    if quotechar is not None:
        dialect.quotechar = quotechar
    if skipinitialspace is not None:
        dialect.skipinitialspace = skipinitialspace

    return dialect


def getstringmetadata(X, metakeys, dialect):

    metadata = {}
    delimiter = dialect.delimiter

    dialist = ['delimiter', 'lineterminator', 'doublequote', 'escapechar', 
               'quoting', 'quotechar','skipinitialspace']


    if 'names' in metakeys:
        _f = tempfile.TemporaryFile('w+b')
        W = csv.writer(_f,delimiter=delimiter,lineterminator='\n')
        W.writerow(X.dtype.names)
        _f.seek(0)
        metadata['names'] = _f.read().strip()
    if 'coloring' in metakeys and X.coloring != {}:
        metadata['coloring'] = repr(X.coloring)
    if 'types' in metakeys:
        metadata['types'] = delimiter.join(parsetypes(X.dtype))
    if 'formats' in metakeys:
        metadata['formats'] = delimiter.join(parseformats(X.dtype))
    if 'dialect' in metakeys:
        diakeys = dialist
    else:
        diakeys = list(set(dialist).intersection(set(metakeys)))
    if len(diakeys) > 0:
        metadata['dialect'] = repr(dict([(a,getattr(dialect,a)) 
                                                             for a in diakeys]))
    
   
    if hasattr(X,'metadata'):
        otherkeys = set(X.metadata.keys()).difference(dialist + 
                               ['names','coloring','types','formats','dialect'])

        for k in otherkeys:
            if k in X.metadata.keys():
                metadata[k] = (X.metadata[k] if is_string_like(X.metadata[k]) 
                               else repr(X.metadata[k]))
            else:
                print 'metadata key', k,'not found'

    return metadata
    

def appendSV(fname, X, **md):
             
    if os.path.exists(fname):
        [md, lines, _w] = getmetadata(fname, inflines=1000, **md)
        if 'names' in md.keys():
            names = md['names']
            if set(names) == set(X.dtype.names):
                F = open(fname,'a')
                Write(X, F, md['dialect'], order=names)
                F.close()
            else:
                raise ValueError, 'names don\'t match:' + str(names) + ', ' + str(X.dtype.names)
        else:
            raise ValueError, 'names can\'t be read'    
    else:
        saveSV(fname,X,**md)
        

def loadbinary(fname):
    """
    Load a numpy binary file or archive created by tabular.io.savebinary.
    
    Load a numpy binary file (``.npy``) or archive (``.npz``) created by 
    :func:`tabular.io.savebinary`.

    The data and associated data type (e.g. `dtype`, including if given, column 
    names) are loaded and reconstituted.

    If `fname` is a numpy archive, it may contain additional data giving 
    hierarchical column-oriented structure (e.g. `coloring`).  See 
    :func:`tabular.tab.tabarray.__new__` for more information about coloring.

    The ``.npz`` file is a zipped archive created using :func:`numpy.savez` and 
    containing one or more ``.npy`` files, which are NumPy binary files created 
    by :func:`numpy.save`.

    **Parameters**

        **fname** : string or file-like object

            File name or open numpy binary file (``.npy``) or archive 
            (``.npz``) created by :func:`tabular.io.savebinary`.

            * When `fname` is a ``.npy`` binary file, it is reconstituted as a 
              flat ndarray of data, with structured dtype.

            * When `fname` is a ``.npz`` archive, it contains at least one 
              ``.npy`` binary file and optionally another:

            * ``data.npy`` must be in the archive, and is reconstituted as `X`, 
              a flat ndarray of data, with structured dtype, `dtype`.

            * ``coloring.npy``, if present is reconstitued as `coloring`, a 
              dictionary.

    **Returns**

        **X** : numpy ndarray with structured dtype

            The data, where each column is named and is of a uniform NumPy data 
            type.

        **dtype** :  numpy dtype object

            The data type of `X`, e.g. `X.dtype`.

        **coloring** :  dictionary, or None

            Hierarchical structure on the columns given in the header of the 
            file; an attribute of tabarrays.

        See :func:`tabular.tab.tabarray.__new__` for more information about 
        coloring.

    **See Also:**

        :func:`tabular.io.savebinary`, :func:`numpy.load`, 
        :func:`numpy.save`, :func:`numpy.savez`

    """

    X = np.load(fname)
    if isinstance(X, np.lib.npyio.NpzFile):
        if 'coloring' in X.files:
            coloring = X['coloring'].tolist()
        else:
            coloring = None
        if 'data' in X.files:
            return [X['data'], X['data'].dtype, coloring]
        else:
            return [None, None, coloring]
    else:
        return [X, X.dtype, None]

def savebinary(fname, X, savecoloring=True):
    """
    Save a tabarray to a numpy binary file or archive.
    
    Save a tabarray to a numpy binary file (``.npy``) or archive
    (``.npz``) that can be loaded by :func:`tabular.io.savebinary`.

    The ``.npz`` file is a zipped archive created using
    :func:`numpy.savez` and containing one or more ``.npy`` files,
    which are NumPy binary files created by :func:`numpy.save`.

    **Parameters**

        **fname** : string or file-like object

            File name or open numpy binary file (``.npy``) or archive (``.npz``) 
            created by :func:`tabular.io.savebinary`.

        **X** :  tabarray

            The actual data in a :class:`tabular.tab.tabarray`:

            * if `fname` is a ``.npy`` file, then this is the same as::

                numpy.savez(fname, data=X)

            * otherwise, if `fname` is a ``.npz`` file, then `X` is zipped 
              inside of `fname` as ``data.npy``

        **savecoloring** : boolean

            Whether or not to save the `coloring` attribute of `X`.  If 
            `savecoloring` is `True`, then `fname` must be a ``.npz`` archive 
            and `X.coloring` is zipped inside of `fname` as ``coloring.npy``

            See :func:`tabular.tab.tabarray.__new__` for more information about 
            coloring.

    **See Also:**

            :func:`tabular.io.loadbinary`, :func:`numpy.load`, 
            :func:`numpy.save`, :func:`numpy.savez`

    """
    if fname[-4:] == '.npy':
        np.save(fname, X)
    else:
        if savecoloring is True:
            np.savez(fname, data=X, coloring=X.coloring)
        else:
            np.savez(fname, data=X)

def inferdelimiterfromname(fname):
    """
    Infer delimiter from file extension.

    * If *fname* ends with '.tsv', return '\\t'.

    * If *fname* ends with '.csv', return ','.

    * Otherwise, return '\\t'.

    **Parameters**

        **fname** :  string

            File path assumed to be for a separated-variable file.

    **Returns**

        **delimiter** :  string

            String in ['\\t', ','], the inferred delimiter.

    """    
    if not is_string_like(fname):
        return '\t'
        
    if fname.endswith('.tsv'):
        return '\t'
    elif fname.endswith('.csv'):
        return ','
    else:
        return '\t'


def parseformats(dtype):
    """
    Parse the formats from a structured numpy dtype object.

    Return list of string representations of numpy formats from a structured 
    numpy dtype object.

    Used by :func:`tabular.io.saveSV` to write out format information in the 
    header.

    **Parameters**

        **dtype** :  numpy dtype object

            Structured numpy dtype object to parse.

    **Returns**

        **out** :  list of strings

            List of strings corresponding to numpy formats::

                [dtype[i].descr[0][1] for i in range(len(dtype))]

    """
    return [dtype[i].descr[0][1] for i in range(len(dtype))]


def parsetypes(dtype):
    """
    Parse the types from a structured numpy dtype object.

    Return list of string representations of types from a structured numpy 
    dtype object, e.g. ['int', 'float', 'str'].

    Used by :func:`tabular.io.saveSV` to write out type information in the 
    header.

    **Parameters**

        **dtype** :  numpy dtype object

            Structured numpy dtype object to parse.

    **Returns**

        **out** :  list of strings

            List of strings corresponding to numpy types::

                [dtype[i].name.strip('1234567890').rstrip('ing') \ 
                 for i in range(len(dtype))]

    """
    return [dtype[i].name.strip('1234567890').rstrip('ing') 
            for i in range(len(dtype))]


def thresholdcoloring(coloring, names):
    """
    Threshold a coloring dictionary for a given list of column names.

    Threshold `coloring` based on `names`, a list of strings in::

        coloring.values()

    **Parameters**

        **coloring** :  dictionary

            Hierarchical structure on the columns given in the header of the 
            file; an attribute of tabarrays.

            See :func:`tabular.tab.tabarray.__new__` for more information about 
            coloring.

        **names** :  list of strings

            List of strings giving column names.

    **Returns**

        **newcoloring** :  dictionary

            The thresholded coloring dictionary.

    """
    for key in coloring.keys():
        if len([k for k in coloring[key] if k in names]) == 0:
            coloring.pop(key)
        else:
            coloring[key] = utils.uniqify([k for k in coloring[key] if k in 
                                           names])
    return coloring

def backslash(dir):
    """
    Add '/' to the end of a path if not already the last character.

    Adds '/' to end of a path (meant to make formatting of directory path `dir` 
    consistently have the slash).

    **Parameters**

        **dir** :  string

            Path to a directory.

    **Returns**

        **out** :  string

            Same as `dir`, with '/' appended to the end if not already there.

    """
    if dir[-1] != '/':
        return dir + '/'
    else:
        return dir

def delete(to_delete):
    """
    Unified "strong" version of delete (remove) for files and directories.

    Unified "strong" version of delete that uses `os.remove` for a file and 
    `shutil.rmtree` for a directory tree.

    **Parameters**

        **to_delete** :  string

            Path to a file or directory.

    **See Also:**

        `os <http://docs.python.org/library/os.html>`_, 
        `shutil <http://docs.python.org/library/shutil.html>`_

    """
    if os.path.isfile(to_delete):
        os.remove(to_delete)
    elif os.path.isdir(to_delete):
        shutil.rmtree(to_delete)

def makedir(dir_name):
    """
     "Strong" directory maker.

    "Strong" version of `os.mkdir`.  If `dir_name` already exists, this deletes 
    it first.

    **Parameters**

        **dir_name** :  string

            Path to a file directory that may or may not already exist.

    **See Also:**

        :func:`tabular.io.delete`, 
        `os <http://docs.python.org/library/os.html>`_

    """
    if os.path.exists(dir_name):
        delete(dir_name)
    os.mkdir(dir_name)
