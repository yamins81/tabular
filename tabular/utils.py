"""
Miscellaneous utilities:  uniqify, listunion, listintersection, perminverse

"""

import numpy as np, traceback
import tabular.fast as fast

__all__ = ['uniqify', 'listunion', 'listintersection', 'perminverse','fromarrays','fromrecords','fromkeypairs','DEFAULT_NULLVALUEFORMAT','DEFAULT_NULLVALUE','DEFAULT_TYPEINFERER']

def uniqify(seq, idfun=None):
    """
    Relatively fast pure Python uniqification function that preservs ordering.

    **Parameters**

            **seq** :  sequence

                    Sequence object to uniqify.

            **idfun** :  function, optional

                    Optional collapse function to identify items as the same.

    **Returns**

            **result** :  list

                    Python list with first occurence of each item in `seq`, in
                    order.

    """
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def listunion(ListOfLists):
    """
    Take the union of a list of lists.

    Take a Python list of Python lists::

            [[l11,l12, ...], [l21,l22, ...], ... , [ln1, ln2, ...]]

    and return the aggregated list::

            [l11,l12, ..., l21, l22 , ...]

    For a list of two lists, e.g. `[a, b]`, this is like::

            a.extend(b)

    **Parameters**

            **ListOfLists** :  Python list

                    Python list of Python lists.

    **Returns**

            **u** :  Python list

                    Python list created by taking the union of the
                    lists in `ListOfLists`.

    """
    u = []
    for s in ListOfLists:
        if s != None:
            u.extend(s)
    return u


def listintersection(ListOfLists):
    u = ListOfLists[0]
    for l in ListOfLists[1:]:
        u = [ll for ll in u if ll in l]
    return u


def perminverse(s):
    '''
    Fast inverse of a (numpy) permutation.

    **Paramters**

            **s** :  sequence

                    Sequence of indices giving a permutation.

    **Returns**

            **inv** :  numpy array

                    Sequence of indices giving the inverse of permutation `s`.

    '''
    X = np.array(range(len(s)))
    X[s] = range(len(s))
    return X


def is_string_like(obj):
    """
    Check whether input object behaves like a string.

    From:  _is_string_like in numpy.lib._iotools

    **Parameters**

        **obj** :  string or file object

                Input object to check.

    **Returns**

        **out** :  bool

                Whether or not `obj` behaves like a string.

    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def listarraytranspose(L):
    '''
    Tranposes the simple array presentation of a list of lists (of equal length).
    Argument:
        L = [row1, row2, ...., rowN]
        where the rowi are python lists of equal length.
    Returns:
        LT, a list of python lists such that LT[j][i] = L[i][j].
    '''
    return [[row[i] for row in L] for i in range(len(L[0]))]


def fromarrays(X, type=None, **kwargs):
    if 'dtype' in kwargs.keys() and kwargs['dtype'] and 'object':
        _array = dtypecolumnloader(X, **kwargs)
    else:
        try:
            _array = np.rec.fromarrays(X, **kwargs)
        except (TypeError,ValueError):
            if 'formats' in kwargs.keys() and kwargs['formats']:
                _array = formatscolumnloader(X, **kwargs)
            else:
                traceback.print_exc()
                raise ValueError
    if type is not None:
        _array = _array.view(type)
    return _array


def fromrecords(X, type=None, **kwargs):
    if 'dtype' in kwargs.keys() and kwargs['dtype']:
        _array = np.array(X,dtype=kwargs['dtype'])
    else:
        try:
            _array = np.rec.fromrecords(X, **kwargs)
        except (TypeError,ValueError):

            if 'formats' in kwargs.keys() and kwargs['formats']:
                formats = processformats(len(X[0]),kwargs['formats'])
                if 'str' not in formats:
                    dtype = getdtype(kwargs['names']
                          if 'names' in kwargs.keys() else None, formats)
                    _array = np.array(X,dtype=dtype)
                else:
                    _array = formatscolumnloader(listarraytranspose(X),
                                   **kwargs)
            else:
                traceback.print_exc()
                raise ValueError
    if type is not None:
        _array = _array.view(type)
    return _array


def fromkeypairs(kvpairs,dtype=None,fillingvalues=None):

    R = np.arange(len(kvpairs))
    KVind = listunion([[(i,) + v for v in kvpairs[i]] \
                               for i in range(len(kvpairs))])
    maxkeyl = max([len(v[1]) for v in KVind]) ; strt = '|S' + str(maxkeyl)
    X = np.array(KVind, dtype = np.dtype([('Row', 'int'),
                                      ('Key', strt), ('Value', 'object')]))
    names = uniqify(X['Key'])

    fvd = processvfd(fillingvalues,names = names)

    realcols = []  ; realtypes = []
    new_version = np.version.short_version >= '1.4.0'
    for (i, n) in enumerate(names):
        Rows = X[X['Key'] == n]['Row']
        Values = X[X['Key'] == n]['Value']

        MissingRows = np.invert(fast.isin(R, Rows)).nonzero()[0]
        Xnr = np.append(Rows, MissingRows)

        if dtype:
            if new_version:
                Xnv = np.array(Values, dtype[n])
            else:
                Xnv = np.array(Values.tolist(),dtype[n])
        else:
            Xnv = np.array(Values.tolist())
        realtype = Xnv.dtype.descr[0][1]
        realtypes.append(realtype)
        fillval = DEFAULT_NULLVALUEFORMAT(realtype) if fvd[i] is None \
               else (fvd[i](Xnv) if hasattr(fvd[i], '__call__') else fvd[i])
        MissingVals = len(MissingRows)*[fillval]
        Xnv = np.append(Xnv,MissingVals)

        s = Xnr.argsort()
        realcols.append(Xnv[s])

    if not dtype:
        dtype = np.dtype(zip(names,realtypes))

    return fromarrays(realcols,dtype=dtype)


def processvfd(fillingvalues, numbers=None, names=None):
    if (numbers or names):
        if not numbers:
            numbers = range(len(names))
        if fillingvalues:
            if isinstance(fillingvalues,list) or \
                           isinstance(fillingvalues,tuple):
               vfd = dict([(n,
                     fillingvalues[i]) for (i,n) in enumerate(numbers)])
            elif isinstance(fillingvalues,dict):
                intkeys = [k for k in fillingvalues.keys() if isinstance(k,int)]
                vfd = dict([(k, v) for (k, v) in fillingvalues.items()
                                 if k in intkeys])
                strkeys = [k for k in fillingvalues.keys() if isinstance(k,str)]
                if len(strkeys) > 0:
                    namedict = dict([(names[j],j) for j in numbers])
                    for n in strkeys:
                        vfd[namedict[n]] = fillingvalues[n]
                for k in numbers:
                    if k not in vfd.keys():
                        vfd[k] = None
            else:
                vfd = dict([(j, fillingvalues) for j in numbers])
        else:
            vfd = dict([(j, None) for j in numbers])
    else:
        vfd = {}
    return vfd


def livetyper(X):
    t = max([type(x) for x in X])
    if t.__name__ == 'str':
        L = max(max([len(x) for x in X]), 1)
        return '|S' + str(L)
    else:
        return np.dtype(t.__name__).descr[0][1]


def DEFAULT_NULLVALUEFORMAT(format):
    """
    Returns a null value for each of the various kinds of numpy formats.

    Default null value function used in :func:`tabular.spreadsheet.join`.

    **Parameters**

            **format** :  string

                    Numpy format descriptor, e.g. ``'<i4'``, ``'|S5'``.

    **Returns**

            **null** :  element in `[0, 0.0, '']`

                    Null value corresponding to the given format:

                    *   if ``format.startswith(('<i', '|b'))``, e.g. `format`
                        corresponds to an integer or Boolean, return 0

                    *   else if `format.startswith('<f')`, e.g. `format`
                        corresponds to a float, return 0.0

                    *   else, e.g. `format` corresponds to a string, return ''

    """
    return 0 if format.startswith(('<i','|b')) \
           else 0.0 if format.startswith('<f') \
           else ''


def DEFAULT_FILLVAL(format):
    return np.nan if format.startswith(('<i','|b','<f')) else ''


def DEFAULT_TYPEINFERER(column):
    """
    Infer the data type (int, float, str) of a list of strings.

    Take a list of strings, and attempts to infer a numeric data type that fits
    them all.

    If the strings are all integers, returns a NumPy array of integers.

    If the strings are all floats, returns a NumPy array of floats.

    Otherwise, returns a NumPy array of the original list of strings.

    Used to determine the datatype of a column read from a separated-variable
    (CSV) text file (e.g. ``.tsv``, ``.csv``) of data where columns are
    expected to be of uniform Python type.

    This function is used by tabular load functions for SV files, e.g. by
    :func`tabular.io.loadSV` when type information is not provided in the
    header, and by :func:`tabular.io.loadSVsafe`.

    **Parameters**

            **column** :  list of strings

                    List of strings corresponding to a column of data.

    **Returns**

            **out** :  numpy array

                    Numpy array of data from `column`, with data type
                    int, float or str.

    """
    try:
        return np.array([int(x) for x in column], 'int')
    except:
        try:
            return np.array([float(x) if x != '' else np.nan for x in column],
                            'float')
        except:
            return np.array(column, 'str')


DEFAULT_STRINGMISSINGVAL = ''


def DEFAULT_STRINGIFIER(D):
    if D.dtype.name.startswith('str'):
        return D
    else:
        return str(D.tolist()).strip('[]').replace('nan', '').split(', ')


def DEFAULT_NULLVALUE(test):
    """
    Returns a null value for each of various kinds of test values.

    **Parameters**

            **test** :  bool, int, float or string

                    Value to test.


    **Returns**
            **null** :  element in `[False, 0, 0.0, '']`

                    Null value corresponding to the given test value:

                    *   if `test` is a `bool`, return `False`
                    *   else if `test` is an `int`, return `0`
                    *   else if `test` is a `float`, return `0.0`
                    *   else `test` is a `str`, return `''`

    """
    return False if isinstance(test,bool) \
           else 0 if isinstance(test,int) \
           else 0.0 if isinstance(test,float) \
           else ''


def dtypecolumnloader(X, **kwargs):
    kk = kwargs.copy()
    dtype = kk['dtype']
    assert len(dtype) == len(X), 'wrong dtypelength'
    return np.rec.fromarrays([makearray(c,dtype[i])
                               for (i, c) in enumerate(X)], **kk)


def formatscolumnloader(X, **kwargs):
    kk = kwargs.copy()
    formats = kk['formats']
    kk.pop('formats')
    formats = processformats(len(X), kwargs['formats'])
    return np.rec.fromarrays([makearray(c, t)
                               for (c, t) in zip(X, formats)], **kk)


def makearray(c, t):
    return np.array(c, t)


def processformats(L, formats):
    if is_string_like(formats):
        formats = formats.split(',')
    if not (len(formats) == 1 or len(formats) == L):
        msg = 'Wrong number of formats (%d) for number of columns (%s)' % \
                                           (len(formats), str(L))
        raise ValueError, msg
    if len(formats) == 1:
        formats = formats*L
    return formats

def getdtype(names, formats):
    return np.dtype(zip(names,formats)) if names != None else np.dtype(formats)
