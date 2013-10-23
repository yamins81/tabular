"""
Spreadsheet-style functions for NumPy ndarray with structured dtype or
recarray objects:

aggregate, aggregate_in, pivot, addrecords, addcols, deletecols, renamecol, 
replace, colstack, rowstack, nullvalue

Note that these functions are also wrapped as methods of the tabular tabarray 
object, which is a subclass of the numpy ndarray.

**See Also:**

        :class:`tabular.tab.tabarray`

"""

__all__ = ['aggregate', 'aggregate_in', 'pivot', 'addrecords', 'addcols', 
           'deletecols', 'renamecol', 'replace', 'colstack', 'rowstack', 
           'join', 'strictjoin', 'DEFAULT_RENAMER']

import numpy as np
import types
import tabular.utils as utils
import tabular.fast as fast
from tabular.colors import GrayScale


def isftype(x):
    a = lambda x : isinstance(x,types.FunctionType) 
    b = isinstance(x,types.BuiltinFunctionType) 
    c = isinstance(x,types.MethodType) 
    d = isinstance(x,types.BuiltinMethodType)
    return a or b or c or d

def aggregate(X, On=None, AggFuncDict=None, AggFunc=None,
              AggList=None, returnsort=False, KeepOthers=True,
              keyfuncdict=None):
    """
    Aggregate a ndarray with structured dtype (or recarray) on columns for 
    given functions.

    Aggregate a numpy recarray (or tabular tabarray) on a set of specified 
    factors, using specified aggregation functions.

    Intuitively, this function will aggregate the dataset `X` on a set of 
    columns, whose names are listed in `On`, so that the resulting aggregate 
    data set has one record for each unique tuples of values in those columns.

    The more factors listed in `On` argument, the "finer" is the aggregation, 
    the fewer factors, the "coarser" the aggregation.  For example, if::

            On = 'A'

    the resulting tabarray will have one record for each unique value of a in 
    X['A'], while if On = ['A', 'B'] then the resulting tabarray will have 
    one record for each unique (a, b) pair in X[['A', 'B']]

    The `AggFunc` argument is a function that specifies how to aggregate the 
    factors _not_ listed in `On`, e.g. the so-called `Off` columns.  For 
    example.  For instance, if On = ['A', 'B'] and `C` is a third column, then ::

            AggFunc = numpy.mean

    will result in a tabarray containing a `C` column whose values are the 
    average of the values from the original `C` columns corresponding to each 
    unique (a, b) pair. 
    
    If you want to specify a different aggreagtion method for each `Off` column,
    use `AggFuncDict` instead of AggFunc.  `AggFuncDict` is a dictionary of
    functions whose keys are column names.  AggFuncDict[C] will be applied to 
    the C column, AggFuncDict[D] to the D column, etc.   AggFunc and AggFuncDict
    can be used simultaneously, with the elements of AggFuncDict overriding 
    AggFunc for the specified columns. 
    
    Using either AggFunc or AggFuncDict, the resulting tabarray has the same 
    columns as the original tabarray.  Sometimes you want to specify the ability
    to create new aggregate columns not corresponding to one specific column in the 
    original tabarray, and taking data from several.  To achieve this, use the
    AggList argument.   AggList is a list of three-element lists of the form:
         (name, func, col_names)
    where `name` specifies the resulting column name in the aggregated tabarray,
    `func` specifies the aggregation function, and `col_names` specifies the 
    list of columns names from the original tabarray that will be needed to 
    compute the aggregate values.  (That is, for each unique tuple `t` in the `On` 
    columns, the subarray of X[col_names] for which X[On] == t is passed to 
    `func`.)
        
    If an `Off` column is _not_ provided as a key in `AggFuncDict`, a default 
    aggregator function will be used:  the sum function for numerical columns, 
    concatenation for string columns.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.aggregate`.

    **Parameters**

            **X** :  numpy ndarray with structured dtype or recarray

                    The data set to aggregate.

            **On** :  string or list of  strings, optional

                    List of column names in `X`.

            **AggFuncDict** :  dictionary, optional

                    Dictionary where

                    *   keys are some (all) column names of `X` that are NOT
                        in `On`

                    *   values are functions that can be applied to lists or
                        numpy arrays.

                    This specifies how to aggregate the factors _not_ listed in
                    `On`, e.g. the so-called `Off` columns.

            **AggFunc** :  function, optional

                    Function that can be applied to lists or numpy arrays,
                    specifying how to aggregate factors not listed in either
                    `On` or the keys of `AggFuncDict`, e.g. a "default"
                    aggregation function for the `Off` columns not explicitly
                    listed in `AggFuncDict`.
                    
            **AggList** :  list, optional

                    List of tuples 

            **returnsort** :        Boolean, optional

                    If `returnsort == True`, then return a list of indices
                    describing how `X` was sorted as a result of aggregation.
                    Default value is `False`.

    **Returns**

            **agg** :  numpy ndarray with structured dtype

                    Aggregated data set.

            **index_array** :  numpy ndarray (int, 1D)

                    Returned only if `returnsort == True`.  List of indices
                    describing how `X` was sorted as a result of aggregation.

    **See also:**

            :func:`tabular.spreadsheet.aggregate_in`


    """
    
    names = X.dtype.names
    if len(X) == 0:
        if returnsort:
            return [X,None]
        else:
            return X

    if On == None:
        On = []
    elif isinstance(On,str):
        On = On.split(',')

    assert all([o in names for o in On]), \
           ("Axes " + str([o for o in On if o not in names]) + 
            " can't be  found.")
   
    if AggList is None:
        AggList = []
    
    if AggFuncDict:
        AggList += AggFuncDict.items()
        
    for (i,x) in enumerate(AggList):
        if utils.is_string_like(x):
            AggList[i] = (x,)
        elif isinstance(x,tuple):
            assert 1 <= len(x) <= 3
            assert isinstance(x[0],str)
            if len(x) == 2 and isinstance(x[1],tuple):
                assert len(x[1]) == 2
                AggList[i] = (x[0],) + x[1]
        else:
            raise ValueError, 'bork'

    Names = [x[0] for x in AggList]
    assert Names == utils.uniqify(Names)
        
    if KeepOthers:
        AggList = [(x,) for x in X.dtype.names if x not in Names + On] + AggList
   
    DefaultChoices = {'string':[], 'sum':[], 'first':[]}

    for (i,v) in enumerate(AggList):
        if len(v) == 1:
            assert v[0] in X.dtype.names
            if AggFunc:
                AggList[i] = v + (AggFunc,v[0])
            else:
                AggList[i] = v + (DefaultChooser(X,v[0], DefaultChoices),v[0])
        elif len(v) == 2:
            if isftype(v[1]):
                assert v[0] in X.dtype.names
                AggList[i] = v + (v[0],)
            elif utils.is_string_like(v[1]):
                if AggFunc:
                    _a = v[1] in X.dtype.names
                    _b = isinstance(v[1],list) and set(v[1]) <= set(X.dtype.names)
                    assert _a or _b
                    AggList[i] = (v[0], AggFunc, v[1])
                else:
                    assert v[1] in X.dtype.names
                    AggList[i] = (v[0],
                                  DefaultChooser(X,v[1],
                                  DefaultChoices),
                                  v[1])
            else:
                raise ValueError,'No specific of name for column.'
        elif len(v) == 3:
            if utils.is_string_like(v[2]):
                assert isftype(v[1]) and v[2] in X.dtype.names
            else:
                assert isftype(v[1]) and \
                (isinstance(v[2],list) and \
                set(v[2]) <= set(X.dtype.names))   

    if len(DefaultChoices['sum']) > 0:
        print('No aggregation function provided for', DefaultChoices['sum'], 
              'so assuming "sum" by default.')
    if len(DefaultChoices['string']) > 0:
        print('No aggregation function provided for', DefaultChoices['string'], 
              'so assuming string concatenation by default.')
    if len(DefaultChoices['first']) > 0:
        print('No aggregation function provided for', DefaultChoices['first'], 
              'and neither summing nor concatenation works, so choosing ' 
              'first value by default.')

    return strictaggregate(X, On, AggList, returnsort, keyfuncdict)        

            
def DefaultChooser(X,o,DC):
    try:
        sum(X[o][0:1])
        DC['sum'].append(o)
        return sum
    except:
        try:
            ''.join(X[o][0:1])
            DC['string'].append(o)
            return ''.join
        except:
            DC['first'].append(o)
            return lambda x : x[0]
            
            
def strictaggregate(X,On,AggList,returnsort=False, keyfuncdict=None):

    if len(On) > 0:
        #if len(On) == 1:
        #    keycols = X[On[0]]
        #else:
        #    keycols = X[On]
        
        keycols = X[On]
        if keyfuncdict is not None:
            for _kf in keyfuncdict:
                fn = keyfuncdict[_kf]
                keycols[_kf] = np.array(map(fn, keycols[_kf]))

        [D, index_array] = fast.recarrayuniqify(keycols)
        X = X[index_array]
        Diffs = np.append(np.append([-1], D[1:].nonzero()[0]), [len(D)])
    else:
        Diffs = np.array([-1, len(X)])

    argcounts = dict([(o,
         f.func_code.co_argcount - (len(f.func_defaults) if \
              f.func_defaults != None else 0) if 'func_code' in dir(f) else 1)
                 for (o,f,g) in AggList])

    OnCols = utils.fromarrays([X[o][Diffs[:-1]+1] for o in On], 
            type=np.ndarray, names=On)
    
    AggColDict = dict([(o,
        [f(X[g][Diffs[i]+1:Diffs[i+1]+1]) if argcounts[o] == 1 else \
        f(X[g][Diffs[i]+1:Diffs[i+1]+1],X) for i in range(len(Diffs) - 1)]) \
        for (o,f,g) in AggList])
    
    if isinstance(AggColDict[AggList[0][0]][0],list) or \
    isinstance(AggColDict[AggList[0][0]][0],np.ndarray):
        lens = map(len, AggColDict[AggList[0][0]])
        OnCols = OnCols.repeat(lens)
        for o in AggColDict.keys():
            AggColDict[o] = utils.listunion(AggColDict[o])
   
    Names = [v[0] for v in AggList]
    AggCols = utils.fromarrays([AggColDict[o] for o in Names], 
           type=np.ndarray, names=Names)
    
    if returnsort:
        return [colstack([OnCols,AggCols]),index_array]
    else:
        return colstack([OnCols,AggCols])


def aggregate_in(Data, On=None, AggFuncDict=None, AggFunc=None, AggList=None,
                  interspersed=True):
    """
    Aggregate a ndarray with structured dtype or recarray
    and include original data in the result.

    Take aggregate of data set on specified columns, then add the resulting 
    rows back into data set to make a composite object containing both original 
    non-aggregate data rows as well as the aggregate rows.

    First read comments for :func:`tabular.spreadsheet.aggregate`.

    This function returns a numpy ndarray, with the number of rows equaling::

            len(Data) + len(A)

    where `A` is the the result of::

            Data.aggregate(On,AggFuncDict)

    `A` represents the aggregate rows; the other rows were the original data 
    rows.

    This function supports _multiple_ aggregation, meaning that one can first 
    aggregate on one set of factors, then repeat aggregation on the result for 
    another set of factors, without the results of the first aggregation 
    interfering the second.  To achieve this, the method adds two new columns:

    *   a column called "__aggregates__" specifying on which factors the rows 
        that are aggregate rows were aggregated.  Rows added by aggregating on 
        factor `A` (a column in the original data set) will have `A` in the 
        "__aggregates__" column.  When multiple factors `A1`, `A2` , ... are 
        aggregated on, the notation is a comma-separated list:  `A1,A2,...`.  
        This way, when you call `aggregate_in` again, the function only 
        aggregates on the columns that have the empty char '' in their 
        "__aggregates__" column.

    *   a column called '__color__', specifying Gray-Scale colors for 
        aggregated rows that will be used by the Data Environment system 
        browser for colorizing the  data.   When there are multiple levels of 
        aggregation, the coarser aggregate groups (e.g. on fewer factors) get 
        darker gray color then those on finer aggregate groups (e.g. more 
        factors).

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.aggregate_in`.

    **Parameters**

            **Data** :  numpy ndarray with structured dtype or recarray

                    The data set to aggregate in.

            **On** :  list of  strings, optional

                    List of column names in `X`.

            **AggFuncDict** :  dictionary, optional

                    Dictionary where

                    *   keys are some (all) column names of `X` that are NOT in 
                        `On`

                    *   values are functions that can be applied to lists or
                        numpy arrays.

                    This specifies how to aggregate the factors _not_ listed in
                    `On`, e.g. the so-called `Off` columns.

            **AggFunc** :  function, optional

                    Function that can be applied to lists or numpy arrays,
                    specifying how to aggregate factors not listed in either 
                    `On` or the keys of `AggFuncDict`, e.g. a "default"
                    aggregation function for the `Off` columns not explicitly
                    listed in `AggFuncDict`.

            **interspersed** :  boolean, optional

                    *   If `True`, aggregate rows are interleaved with the data 
                        of which they are aggregates.

                    *   If `False`, all aggregate rows placed at the end of the 
                        array.

    **Returns**

            **agg** :  numpy ndarray with structured dtype

                    Composite aggregated data set plus original data set.

    **See also:**

            :func:`tabular.spreadsheet.aggregate`

    """

    # See if there's an '__aggregates__ column'.  
    # If so, strip off all those that are nontrivial.

    Data = deletecols(Data,'__color__')
    if '__aggregates__' in Data.dtype.names:
        X = Data[Data['__aggregates__'] == ''][:]
        OldAggregates = Data[Data['__aggregates__'] != ''][:]
        AggVars = utils.uniqify(utils.listunion([x.split(',') for x in 
                                OldAggregates['__aggregates__']]))
    else:
        X = Data
        OldAggregates = Data[0:0]
        AggVars = []

    if On == None:
        On = []

    NewAggregates = aggregate(X, On, AggFuncDict=AggFuncDict, 
                AggFunc=AggFunc, AggList=AggList, KeepOthers=True)
    on = ','.join(On)
    NewAggregates = addcols(NewAggregates,   
                            utils.fromarrays([[on]*len(NewAggregates)], 
                            type=np.ndarray, names=['__aggregates__']))
    AggVars = utils.uniqify(AggVars + On)
    Aggregates = rowstack([OldAggregates,NewAggregates],mode='nulls')

    ANLen = np.array([len(x.split(',')) for x in Aggregates['__aggregates__']])
    U = np.array(utils.uniqify(ANLen)); U.sort()
    [A,B] = fast.equalspairs(ANLen,U)
    Grays = np.array(grayspec(len(U)))
    AggColor = utils.fromarrays([Grays[A]], type=np.ndarray, 
           names = ['__color__'])

    Aggregates = addcols(Aggregates,AggColor)

    if not interspersed or len(AggVars) == 0:
        return rowstack([X,Aggregates],mode='nulls')
    else:
        s = ANLen.argsort()
        Aggregates = Aggregates[s[range(len(Aggregates) - 1, -1, -1)]]
        X.sort(order = AggVars)
        Diffs = np.append(np.append([0], 1 + (X[AggVars][1:] != 
                                    X[AggVars][:-1]).nonzero()[0]), [len(X)])
        DiffAtts = ([[t for t in AggVars if X[t][Diffs[i]] != X[t][Diffs[i+1]]] 
                      for i in range(len(Diffs) - 2)] 
                     if len(Diffs) > 2 else []) + [AggVars]

        HH = {}
        for l in utils.uniqify(Aggregates['__aggregates__']):
            Avars = l.split(',')
            HH[l] = fast.recarrayequalspairs(X[Avars][Diffs[:-1]], 
                                             Aggregates[Avars])

        Order = []
        for i in range(len(Diffs)-1):
            Order.extend(range(Diffs[i], Diffs[i+1]))

            Get = []
            for l in HH.keys():
                Get += [len(X) + j for j in 
                        HH[l][2][range(HH[l][0][i], HH[l][1][i])] if 
                        len(set(DiffAtts[i]).intersection(
                        Aggregates['__aggregates__'][j].split(','))) > 0 and 
                        set(Aggregates['__aggregates__'][j].split(',')) == 
                        set(l.split(','))]

            Order.extend(Get)

        return rowstack([X, Aggregates], mode='nulls')[Order]


def grayspec(k):
    """
    List of gray-scale colors in HSV space as web hex triplets.

    For integer argument k, returns list of `k` gray-scale colors, increasingly 
    light, linearly in the HSV color space, as web hex triplets.

    Technical dependency of :func:`tabular.spreadsheet.aggregate_in`.

    **Parameters**

            **k** :  positive integer

                    Number of gray-scale colors to return.

    **Returns**

            **glist** :  list of strings

                    List of `k` gray-scale colors.

    """
    ll = .5
    ul = .8
    delta = (ul - ll) / k
    return [GrayScale(t) for t in np.arange(ll, ul, delta)]


def pivot(X, a, b, Keep=None, NullVals=None, order = None, prefix='_'):
    '''
    Implements pivoting on numpy ndarrays (with structured dtype) or recarrays.

    See http://en.wikipedia.org/wiki/Pivot_table for information about pivot 
    tables.

    Returns `X` pivoted on (a,b) with `a` as the row axis and `b` values as the 
    column axis.

    So-called "nontrivial columns relative to `b`" in `X` are added as 
    color-grouped sets of columns, and "trivial columns relative to `b`" are 
    also retained as cross-grouped sets of columns if they are listed in `Keep` 
    argument.

    Note that a column `c` in `X` is "trivial relative to `b`" if for all rows 
    i, X[c][i] can be determined from X[b][i], e.g the elements in X[c] are in 
    many-to-any correspondence with the values in X[b].

    The function will raise an exception if the list of pairs of value in 
    X[[a,b]] is not the product of the individual columns values, e.g.::

            X[[a,b]] == set(X[a]) x set(X[b])

    in some ordering.

    Implemented by the tabarray method :func:`tabular.tab.tabarray.pivot`

    **Parameters**

            **X** :  numpy ndarray with structured dtype or recarray

                    The  data set to pivot.

            **a** : string

                    Column name in `X`.

            **b** : string

                    Another column name in `X`.

            **Keep** :  list of strings, optional

                    List of other columns names in `X`.

            **NullVals** :  optional

                    Dictionary mapping column names in `X` other than `a` or 
                    `b` to appropriate null values for their types.

                    If `None`, then the null values defined by the `nullvalue`
                    function are used, see
                    :func:`tabular.spreadsheet.nullvalue`.

            **prefix** :  string, optional

                    Prefix to add to `coloring` keys corresponding to 
                    cross-grouped "trivial columns relative to `b`".  Default 
                    value is an underscore, '_'.

    **Returns**

            **ptable** :  numpy ndarray with structured dtype

                    The resulting pivot table.

            **coloring** :  dictionary

                    Dictionary whose keys are strings and corresponding values 
                    are lists of column names (e.g. strings).

                    There are two groups of keys:

                    *   So-called "nontrivial columns relative to `b`" in `X`.  
                        These correspond to columns in::

                                    set(`X.dtype.names`) - set([a, b])

                    *   Cross-grouped "trivial columns relative to `b`".  The 
                        `prefix` is used to distinguish these.

                    The `coloring` parameter is used by the the tabarray pivot 
                    method, :func:`tabular.tab.tabarray.pivot`.

                    See :func:`tabular.tab.tabarray.__new__` for more
                    information about coloring.

    '''

    othernames = [o for o in X.dtype.names if o not in [a,b]]

    for c in [a,b]:
        assert c in X.dtype.names, 'Column ' + c + ' not found.'

    [D,s] = fast.recarrayuniqify(X[[a,b]])
    unique_ab = X[[a,b]][s[D.nonzero()[0]]]
    assert len(X) == len(unique_ab) , \
           ('Pairs of values in columns', a, 'and', b, 'must be unique.')

    [D,s] = fast.arrayuniqify(X[a])
    unique_a = X[a][s[D.nonzero()[0]]]
    [D,s] = fast.arrayuniqify(X[b])
    unique_b = X[b][s[D.nonzero()[0]]]

    Da = len(unique_a)
    Db = len(unique_b)

    if len(X) != Da * Db:
        if list(X.dtype.names).index(a) < list(X.dtype.names).index(b):
            n1 = a ; f1 = unique_a; n2 = b ; f2 = unique_b
        else:
            n1 = b ; f1 = unique_b; n2 = a ; f2 = unique_a
            
        dtype = np.dtype([(n1,f1.dtype.descr[0][1]),(n2,f2.dtype.descr[0][1])])
        allvalues = utils.fromarrays([np.repeat(f1,
              len(f2)),
              np.tile(f2,len(f1))],
              np.ndarray,
              dtype=dtype)
        
        
        missingvalues = allvalues[np.invert(fast.recarrayisin(allvalues,
                                      X[[a,b]]))]

        if NullVals == None:
           NullVals = {}
        
        if not isinstance(NullVals,dict):
            if hasattr(NullVals,'__call__'):
                NullVals = dict([(o,NullVals(o)) for o in othernames])
            else:
                NullVals = dict([(o,NullVals) for o in othernames])

        nullvals = utils.fromrecords([[NullVals[o] if o in NullVals.keys() 
                            else utils.DEFAULT_NULLVALUE(X[o][0]) for o in 
                            othernames]], type=np.ndarray, names=othernames)
        
        nullarray = nullvals.repeat(len(missingvalues))
        Y = colstack([missingvalues, nullarray])
        Y = Y.astype(np.dtype([(o,
                  X.dtype[o].descr[0][1]) for o in Y.dtype.names]))
        X = rowstack([X, Y])

    X.sort(order = [a,b])

    Bvals = X[b][:Db]
    bnames = [str(bv).replace(' ','') for bv in Bvals]

    assert (len(set(othernames).intersection(bnames)) == 0 and 
            a not in bnames), ('Processed values of column', b, 
                               'musn\'t intersect with other column names.')

    acol = X[a][::Db]

    Cols = [acol]
    names = [a]
    Trivials = []
    NonTrivials = []
    for c in othernames:
        Z = X[c].reshape((Da,Db))

        if all([len(set(Z[:,i])) == 1 for i in range(Z.shape[1])]):
            Trivials.append(c)
        else:
            NonTrivials.append(c)
        Cols += [Z[:,i] for i in range(Z.shape[1])]
        names += [bn + '_' + c for bn in bnames]
    
    if order is not None:
        ordering = [names.index(ord) for ord in order]
        Cols = [Cols[i] for i in ordering]
        names = [names[i] for i in ordering]
    dtype = np.dtype([(n,c.dtype.descr[0][1]) for (n,c) in zip(names,Cols)])
    D = utils.fromarrays(Cols,type=np.ndarray,dtype=dtype)

    coloring = {}
    if Keep != None:
        Trivials = set(Trivials).intersection(Keep)
        for c in Trivials:
            X.sort(order=[c])
            cvals = np.array(utils.uniqify(X[c]))
            [AA,BB] = fast.equalspairs(cvals,X[c])

            for (i,cc) in enumerate(cvals):
                blist = [str(bv).replace(' ', '') for bv in Bvals if bv in 
                         X[b][AA[i]:BB[i]]]
                coloring[str(cc)] = [a] + [bn + '_' + d for bn in blist for d 
                                           in NonTrivials]
                for d in NonTrivials:
                    coloring[str(cc) + '_' + d] = [a] + blist

    for c in NonTrivials:
        coloring[c] = [a] + [bn + '_' + c for bn in bnames]
    for bn in bnames:
        coloring[prefix + bn] = [a] + [bn + '_' + c for c in NonTrivials]

    return [D, coloring]
        
def addrecords(X, new):
    """
    Append one or more records to the end of a numpy recarray or ndarray .

    Can take a single record, void or tuple, or a list of records, voids or 
    tuples.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.addrecords`.

    **Parameters**

            **X** :  numpy ndarray with structured dtype or recarray

                    The array to add records to.

            **new** :  record, void or tuple, or list of them

                    Record(s) to add to `X`.

    **Returns**

            **out** :  numpy ndarray with structured dtype

                    New numpy array made up of `X` plus the new records.

    **See also:**  :func:`tabular.spreadsheet.rowstack`

    """
    if isinstance(new, np.record) or isinstance(new, np.void) or \
                                                        isinstance(new, tuple):
        new = [new]
    return np.append(X, utils.fromrecords(new, type=np.ndarray,
                                              dtype=X.dtype), axis=0)


def addcols(X, cols, names=None):
    """
    Add one or more columns to a numpy ndarray.

    Technical dependency of :func:`tabular.spreadsheet.aggregate_in`.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.addcols`.

    **Parameters**

            **X** :  numpy ndarray with structured dtype or recarray

                    The recarray to add columns to.

            **cols** :  numpy ndarray, or list of arrays of columns
            
                    Column(s) to add.

            **names**:  list of strings, optional

                    Names of the new columns. Only applicable when `cols` is a 
                    list of arrays.

    **Returns**

            **out** :  numpy ndarray with structured dtype

                    New numpy array made up of `X` plus the new columns.

    **See also:**  :func:`tabular.spreadsheet.colstack`

    """

    if isinstance(names,str):
        names = [n.strip() for n in names.split(',')]

    if isinstance(cols, list):
        if any([isinstance(x,np.ndarray) or isinstance(x,list) or \
                                     isinstance(x,tuple) for x in cols]):
            assert all([len(x) == len(X) for x in cols]), \
                   'Trying to add columns of wrong length.'
            assert names != None and len(cols) == len(names), \
                   'Number of columns to add must equal number of new names.'
            cols = utils.fromarrays(cols,type=np.ndarray,names = names)
        else:
            assert len(cols) == len(X), 'Trying to add column of wrong length.'
            cols = utils.fromarrays([cols], type=np.ndarray,names=names)
    else:
        assert isinstance(cols, np.ndarray)
        if cols.dtype.names == None:
            cols = utils.fromarrays([cols],type=np.ndarray, names=names)

    Replacements = [a for a in cols.dtype.names if a in X.dtype.names]
    if len(Replacements) > 0:
        print('Replacing columns', 
              [a for a in cols.dtype.names if a in X.dtype.names])

    return utils.fromarrays(
      [X[a] if a not in cols.dtype.names else cols[a] for a in X.dtype.names] + 
      [cols[a] for a in cols.dtype.names if a not in X.dtype.names], 
      type=np.ndarray,
      names=list(X.dtype.names) + [a for a in cols.dtype.names 
                                   if a not in X.dtype.names])


def deletecols(X, cols):
    """
    Delete columns from a numpy ndarry or recarray.

    Can take a string giving a column name or comma-separated list of column 
    names, or a list of string column names.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.deletecols`.

    **Parameters**

            **X** :  numpy recarray or ndarray with structured dtype

                    The numpy array from which to delete columns.

            **cols** :  string or list of strings

                    Name or list of names of columns in `X`.  This can be
                    a string giving a column name or comma-separated list of 
                    column names, or a list of string column names.

    **Returns**

            **out** :  numpy ndarray with structured dtype

                    New numpy ndarray with structured dtype
                    given by `X`, excluding the columns named in `cols`.

    """
    if isinstance(cols, str):
        cols = cols.split(',')
    retain = [n for n in X.dtype.names if n not in cols]
    if len(retain) > 0:
        return X[retain]
    else:
        return None


def renamecol(X, old, new):
    """
    Rename column of a numpy ndarray with structured dtype, in-place.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.renamecol`.

    **Parameters**

            **X** :  numpy ndarray with structured dtype

                    The numpy array for which a column is to be renamed.

            **old** :  string

                    Old column name, e.g. a name in `X.dtype.names`.

            **new** :  string

                    New column name to replace `old`.

    """
    NewNames = tuple([n if n != old else new for n in X.dtype.names])
    X.dtype.names = NewNames


def replace(X, old, new, strict=True, cols=None, rows=None):
    """
    Replace value `old` with `new` everywhere it appears in-place.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.replace`.

    **Parameters**

            **X** :  numpy ndarray with structured dtype

                    Numpy array for which in-place replacement of `old` with 
                    `new` is to be done.

            **old** : string

            **new** : string

            **strict** :  boolean, optional

            *   If `strict` = `True`, replace only exact occurences of `old`.

            *   If `strict` = `False`, assume `old` and `new` are strings and   
                replace all occurences of substrings (e.g. like 
                :func:`str.replace`)

            **cols** :  list of strings, optional

                    Names of columns to make replacements in; if `None`, make 
                    replacements everywhere.

            **rows** : list of booleans or integers, optional

                    Rows to make replacements in; if `None`, make replacements 
                    everywhere.

    Note:  This function does in-place replacements.  Thus there are issues 
    handling data types here when replacement dtype is larger than original 
    dtype.  This can be resolved later by making a new array when necessary ...

    """

    if cols == None:
        cols = X.dtype.names
    elif isinstance(cols, str):
        cols = cols.split(',')

    if rows == None:
        rows = np.ones((len(X),), bool)

    if strict:
        new = np.array(new)
        for a in cols:
            if X.dtype[a] < new.dtype:
                print('WARNING: dtype of column', a, 
                      'is inferior to dtype of ', new, 
                      'which may cause problems.')
            try:
                X[a][(X[a] == old)[rows]] = new
            except:
                print('Replacement not made on column', a, '.')
    else:
        for a in cols:
            QuickRep = True
            try:
                colstr = ''.join(X[a][rows])
            except TypeError:
                print('Not replacing in column', a, 'due to type mismatch.')
            else:
                avoid = [ord(o) for o in utils.uniqify(old + new + colstr)]
                ok = set(range(256)).difference(avoid)
                if len(ok) > 0:
                    sep = chr(list(ok)[0])
                else:
                    ok = set(range(65536)).difference(avoid)
                    if len(ok) > 0:
                        sep = unichr(list(ok)[0])
                    else:
                        print('All unicode characters represented in column', 
                              a, ', can\t replace quickly.')
                        QuickRep = False

                if QuickRep:
                    newrows = np.array(sep.join(X[a][rows])
                                       .replace(old, new).split(sep))
                else:
                    newrows = np.array([aa.replace(old,new) for aa in 
                                        X[a][rows]])
                X[a][rows] = np.cast[X.dtype[a]](newrows)

                if newrows.dtype > X.dtype[a]:
                    print('WARNING: dtype of column', a, 'is inferior to the ' 
                          'dtype of its replacement which may cause problems '
                          '(ends of strings might get chopped off).')


def rowstack(seq, mode='nulls', nullvals=None):
    '''
    Vertically stack a sequence of numpy ndarrays with structured dtype

    Analog of numpy.vstack

    Implemented by the tabarray method
    :func:`tabular.tab.tabarray.rowstack` which uses 
    :func:`tabular.tabarray.tab_rowstack`.

    **Parameters**

            **seq** :  sequence of numpy recarrays

                    List, tuple, etc. of numpy recarrays to stack vertically.

            **mode** :  string in ['nulls', 'commons', 'abort']

                    Denotes how to proceed if the recarrays have different
                    dtypes, e.g. different sets of named columns.

                    *   if `mode` == ``nulls``, the resulting set of columns is
                        determined by the union of the dtypes of all recarrays
                        to be stacked, and missing data is filled with null 
                        values as defined by 
                        :func:`tabular.spreadsheet.nullvalue`; this is the 
                        default mode.

                    *   elif `mode` == ``commons``, the resulting set of 
                        columns is determined by the intersection of the dtypes 
                        of all recarrays to be stacked, e.g. common columns.

                    *   elif `mode` == ``abort``, raise an error when the
                        recarrays to stack have different dtypes.

    **Returns**

            **out** :  numpy ndarray with structured dtype

                    Result of vertically stacking the arrays in `seq`.

    **See also:**  `numpy.vstack 
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html>`_.

    '''

    if nullvals == None:
        nullvals = utils.DEFAULT_NULLVALUEFORMAT
    #newseq = [ss for ss in seq if len(ss) > 0]
    if len(seq) > 1:
        assert mode in ['commons','nulls','abort'], \
             ('"mode" argument must either by "commons", "abort", or "nulls".')
        if mode == 'abort':
            if not all([set(l.dtype.names) == set(seq[0].dtype.names) 
                        for l in seq]):
                raise ValueError('Some column names are different.')
            else:
                mode = 'commons'
        if mode == 'nulls':
            names =  utils.uniqify(utils.listunion([list(s.dtype.names) 
                                       for s in seq if s.dtype.names != None]))
            formats = [max([s.dtype[att] for s in seq if s.dtype.names != None 
                       and att in s.dtype.names]).str for att in names]         
            dtype = np.dtype(zip(names,formats))
            return utils.fromarrays([utils.listunion([s[att].tolist() 
                        if (s.dtype.names != None and att in s.dtype.names) 
                        else [nullvals(format)] * len(s) for s in seq]) 
                        for (att, format) in zip(names, formats)], type=np.ndarray,
                        dtype=dtype)
        elif mode == 'commons':
            names = [x for x in seq[0].dtype.names 
                     if all([x in l.dtype.names for l in seq[1:]])]
            formats = [max([a.dtype[att] for a in seq]).str for att in names]
            return utils.fromrecords(utils.listunion(
                    [ar.tolist() for ar in seq]), type=np.ndarray,
                          names=names, formats=formats)
    else:
        return seq[0]


def colstack(seq, mode='abort',returnnaming=False):
    """
    Horizontally stack a sequence of numpy ndarrays with structured dtypes

    Analog of numpy.hstack for recarrays.

    Implemented by the tabarray method 
    :func:`tabular.tab.tabarray.colstack` which uses 
    :func:`tabular.tabarray.tab_colstack`.

    **Parameters**

            **seq** :  sequence of numpy ndarray with structured dtype

                    List, tuple, etc. of numpy recarrays to stack vertically.

            **mode** :  string in ['first','drop','abort','rename']

                    Denotes how to proceed if when multiple recarrays share the 
                    same column name:

                    *   if `mode` == ``first``, take the column from the first
                        recarray in `seq` containing the shared column name.

                    *   elif `mode` == ``abort``, raise an error when the 
                        recarrays to stack share column names; this is the
                        default mode.

                    *   elif `mode` == ``drop``, drop any column that shares    
                        its name with any other column among the sequence of 
                        recarrays.

                    *   elif `mode` == ``rename``, for any set of all columns
                        sharing the same name, rename all columns by appending 
                        an underscore, '_', followed by an integer, starting 
                        with '0' and incrementing by 1 for each subsequent 
                        column.

    **Returns**

            **out** :  numpy ndarray with structured dtype

                    Result of horizontally stacking the arrays in `seq`.

    **See also:**  `numpy.hstack 
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html>`_.

    """
    assert mode in ['first','drop','abort','rename'], \
       'mode argument must take on value "first","drop", "rename", or "abort".'

    AllNames = utils.uniqify(utils.listunion(
                                           [list(l.dtype.names) for l in seq]))
    NameList = [(x, [i for i in range(len(seq)) if x in seq[i].dtype.names]) 
                     for x in AllNames]
    Commons = [x[0] for x in NameList if len(x[1]) > 1]

    if len(Commons) > 0 or mode == 'first':
        if mode == 'abort':
            raise ValueError('There are common column names with differing ' +              
                             'values in the columns')
        elif mode == 'drop':
            Names = [(L[0], x,x) for (x, L) in NameList if x not in Commons]
        elif mode == 'rename':
            NameDict = dict(NameList)
            Names = utils.listunion([[(i,n,n) if len(NameDict[n]) == 1 else \
               (i,n,n + '_' + str(i)) for n in s.dtype.names] \
                                   for (i,s) in enumerate(seq)])                           
    else:
        Names = [(L[0], x,x) for (x, L) in NameList]
    
    if returnnaming:
        return  utils.fromarrays([seq[i][x] for (i, x,y) in Names], 
                 type= np.ndarray,names=zip(*Names)[2]),Names
    else:
        return utils.fromarrays([seq[i][x] for (i, x,y) in Names], 
                 type= np.ndarray,names=zip(*Names)[2])


def join(L, keycols=None, nullvals=None, renamer=None, 
         returnrenaming=False, Names=None):
    """
    Combine two or more numpy ndarray with structured dtype on common key 
    column(s).

    Merge a list (or dictionary) of numpy ndarray with structured dtype, given
    by `L`, on key columns listed in `keycols`.

    This function is actually a wrapper for 
    :func:`tabular.spreadsheet.strictjoin`.

    The ``strictjoin`` function has a few restrictions, and this ``join``
    function will try to ensure that they are satisfied:

    *   each element of `keycol` must be a valid column name in `X`
        and each array in `L`, and all of the same data-type.

    *   for each column `col`  in `keycols`, and each array `A` in `L`, the 
        values in `A[col]` must be unique, -- and same for `X[col]`.  
        (Actually this uniqueness doesn't have to hold for the first tabarray
        in L, that is, L[0], but must for all the subsequent ones.)

    *   the *non*-key-column column names in each of the arrays must be 
        disjoint from each other -- or disjoint after a renaming (see below).

    An error will be thrown if these conditions are not met.

    If you don't provide a value of `keycols`, the algorithm will attempt to 
    infer which columns should be used by trying to find the largest set of 
    common column names that contain unique values in each array and have the 
    same data type.  An error will be thrown if no such inference can be made.

    *Renaming of overlapping columns*

            If the non-keycol column names of the arrays overlap, ``join`` will 
            by default attempt to rename the columns by using a simple 
            convention:

            *   If `L` is a list, it will append the number in the list to the 
                key associated with the array.

            *   If `L` is a dictionary, the algorithm will append the string 
                representation of the key associated with an array to the 
                overlapping columns from that array.

            You can override the default renaming scheme using the `renamer` 
            parameter.

    *Nullvalues for keycolumn differences*

            If there are regions of the keycolumns that are not overlapping 
            between merged arrays, `join` will fill in the relevant entries 
            with null values chosen by default:

            *   '0' for integer columns

            *   '0.0' for float columns

            *   the empty character ('') for string columns.

    **Parameters**

            **L** :  list or dictionary

                    Numpy recarrays to merge.  If `L` is a dictionary, the keys
                    name each numpy recarray, and the corresponding values are 
                    the actual numpy recarrays.

            **keycols** :  list of strings

                    List of the names of the key columns along which to do the 
                    merging.

            **nullvals** :  function, optional

                    A function that returns a null value for a numpy format
                    descriptor string, e.g. ``'<i4'`` or ``'|S5'``.

                    See the default function for further documentation:

                            :func:`tabular.spreadsheet.DEFAULT_NULLVALUEFORMAT`

            **renamer** :  function, optional

                    A function for renaming overlapping non-key column names 
                    among the numpy recarrays to merge.

                    See the default function for further documentation:

                            :func:`tabular.spreadsheet.DEFAULT_RENAMER`

            **returnrenaming** :  Boolean, optional

                    Whether to return the result of the `renamer` function.

                    See the default function for further documentation:

                            :func:`tabular.spreadsheet.DEFAULT_RENAMER`

            **Names**: list of strings:

                    If `L` is a list, than names for elements of `L` can be 
                    specified with `Names` (without losing the ordering as you 
                    would if you did it with a dictionary).  
                    
                    `len(L)` must equal `len(Names)`

    **Returns**

            **result** :  numpy ndarray with structured dtype

                    Result of the join, e.g. the result of merging the input
                    numpy arrays defined in `L` on the key columns listed in 
                    `keycols`.

            **renaming** :  dictionary of dictionaries, optional

                    The result returned by the `renamer` function. Returned 
                    only if `returnrenaming == True`.

                    See the default function for further documentation:

                            :func:`tabular.spreadsheet.DEFAULT_RENAMER`

    **See Also:**

            :func:`tabular.spreadsheet.strictjoin`

    """

    if isinstance(L, dict):
        Names = L.keys()
        LL = L.values()
    else:
        if Names == None:
            Names = range(len(L))
        else:
            assert len(Names) == len(L)
        LL = L

    if not keycols:
        keycols = utils.listintersection([a.dtype.names for a in LL])
        if len(keycols) == 0:
            raise ValueError('No common column names found.')

        keycols = [l for l in keycols if all([a.dtype[l] == LL[0].dtype[l] 
                   for a in LL])]
        if len(keycols) == 0:
            raise ValueError('No suitable common keycolumns, '
                             'with identical datatypes found.')

        keycols = [l for l in keycols if all([isunique(a[keycols]) 
                   for a in LL])]
        if len(keycols) == 0:
            raise ValueError('No suitable common keycolumns, '
                             'with unique value sets in all arrays to be '
                             'merged, were found.')
        else:
            print('Inferring keycols to be:', keycols)

    elif isinstance(keycols,str):
        keycols = [l.strip() for l in keycols.split(',')]

    commons = set(Commons([l.dtype.names for l in LL])).difference(keycols)
    renaming = {}
    if len(commons) > 0:
        print 'common attributes, forcing a renaming ...'
        if renamer == None:
            print('Using default renamer ...')
            renamer = DEFAULT_RENAMER
        renaming = renamer(L, Names=Names)
        if not RenamingIsInCorrectFormat(renaming, L, Names=Names):
            print('Renaming from specified renamer is not in correct format,'
                  'using default renamer instead ...')
            renaming = DEFAULT_RENAMER(L, Names = Names)
        NewNames = [[l if l not in renaming[k].keys() else renaming[k][l] 
                     for l in ll.dtype.names] for (k, ll) in zip(Names, LL)]
        if set(Commons(NewNames)).difference(keycols):
            raise ValueError('Renaming convention failed to produce '
                             'separated names.')

    Result = strictjoin(L, keycols, nullvals, renaming, Names=Names)

    if returnrenaming:
        return [Result, renaming]
    else:
        if renaming:
            print('There was a nontrivial renaming, to get it set '      
                  '"returnrenaming = True" in keyword to join function.')
        return Result


def strictjoin(L, keycols, nullvals=None, renaming=None, Names=None):
    """
    Combine two or more numpy ndarray with structured dtypes on common key
    column(s).

    Merge a list (or dictionary) of numpy arrays, given by `L`, on key 
    columns listed in `keycols`.

    The ``strictjoin`` assumes the following restrictions:

    *   each element of `keycol` must be a valid column name in `X` and each 
        array in `L`, and all of the same data-type.

    *   for each column `col`  in `keycols`, and each array `A` in `L`, the 
        values in `A[col]` must be unique, e.g. no repeats of values -- and 
        same for `X[col]`.  (Actually, the uniqueness criterion need not hold
        to the first tabarray in L, but first for all the subsequent ones.)

    *   the *non*-key-column column names in each of the arrays must be 
        disjoint from each other -- or disjoint after a renaming (see below).

    An error will be thrown if these conditions are not met.

    For a wrapper that attempts to meet these restrictions, see 
    :func:`tabular.spreadsheet.join`.

    If you don't provide a value of `keycols`, the algorithm will attempt to 
    infer which columns should be used by trying to find the largest set of 
    common column names that contain unique values in each array and have the 
    same data type.  An error will be thrown if no such inference can be made.

    *Renaming of overlapping columns*

            If the non-keycol column names of the arrays overlap, ``join`` will 
            by default attempt to rename the columns by using a simple 
            convention:

            *   If `L` is a list, it will append the number in the list to the 
                key associated with the array.

            *   If `L` is a dictionary, the algorithm will append the string 
                representation of the key associated with an array to the 
                overlapping columns from that array.

            You can override the default renaming scheme using the `renamer` 
            parameter.

    *Nullvalues for keycolumn differences*

            If there are regions of the keycolumns that are not overlapping 
            between merged arrays, `join` will fill in the relevant entries 
            with null values chosen by default:

            *   '0' for integer columns

            *   '0.0' for float columns

            *   the empty character ('') for string columns.

    **Parameters**

            **L** :  list or dictionary

                    Numpy recarrays to merge.  If `L` is a dictionary, the keys
                    name each numpy recarray, and the corresponding values are 
                    the actual numpy recarrays.

            **keycols** :  list of strings

                    List of the names of the key columns along which to do the 
                    merging.

            **nullvals** :  function, optional

                    A function that returns a null value for a numpy format
                    descriptor string, e.g. ``'<i4'`` or ``'|S5'``.

                    See the default function for further documentation:

                            :func:`tabular.spreadsheet.DEFAULT_NULLVALUEFORMAT`

            **renaming** :  dictionary of dictionaries, optional

                    Dictionary mapping each input numpy recarray to a 
                    dictionary mapping each original column name to its new 
                    name following the convention above.

                    For example, the result returned by:

                            :func:`tabular.spreadsheet.DEFAULT_RENAMER`

    **Returns**

            **result** :  numpy ndarray with structured dtype

                    Result of the join, e.g. the result of merging the input
                    numpy arrays defined in `L` on the key columns listed in 
                    `keycols`.

    **See Also:**

            :func:`tabular.spreadsheet.join`
    """

    if isinstance(L,dict):
        Names = L.keys()
        LL = L.values()
    else:
        if Names == None:
            Names = range(len(L))
        else:
            assert len(Names) == len(L)
        LL = L

    if isinstance(keycols,str):
        keycols = [l.strip() for l in keycols.split(',')]

    assert all([set(keycols) <= set(l.dtype.names) for l in LL]), \
           ('keycols,', str(keycols), 
            ', must be valid column names in all arrays being merged.')
    assert all([isunique(l[keycols]) for l in LL[1:]]), \
           ('values in keycol columns,', str(keycols), 
            ', must be unique in all arrays being merged.')

    if renaming == None:
        renaming = {}
    assert RenamingIsInCorrectFormat(renaming, L, Names=Names), \
           'renaming is not in proper format ... '

    L = dict([(k,ll.copy()) for (k,ll) in zip(Names,LL)])
    LL = L.values()

    for i in Names:
        l = L[i]
        l.dtype = np.dtype(l.dtype.descr)
        if i in renaming.keys():
            for k in renaming[i].keys():
                if k not in keycols:
                    renamecol(L[i], k, renaming[i][k])
        l.sort(order = keycols)


    commons = set(Commons([l.dtype.names for l in LL])).difference(keycols)
    assert len(commons) == 0, ('The following (non-keycol) column names ' 
                    'appear in more than on array being merged:', str(commons))

    Result = colstack([(L[Names[0]][keycols])[0:0]] + 
                      [deletecols(L[k][0:0], keycols) \
                      for k in Names if deletecols(L[k][0:0], keycols) != None])

    PL = powerlist(Names)
    ToGet = utils.listunion([[p for p in PL if len(p) == k] 
                             for k in range(1, len(Names))]) + [PL[-1]]

    for I in ToGet[::-1]:
        Ref = L[I[0]][keycols]

        for j in I[1:]:
            if len(Ref) > 0:
                Ref = Ref[fast.recarrayisin(Ref, L[j][keycols], weak=True)]
            else:
                break

        if len(Ref) > 0:
            D = [fast.recarrayisin(L[j][keycols], Ref, weak=True) for j in I]
            
            Ref0 = L[I[0]][keycols][D[0]]
            Reps0 = np.append(np.append([-1],
                (Ref0[1:] != Ref0[:-1]).nonzero()[0]),[len(Ref0)-1])
            Reps0 = Reps0[1:] - Reps0[:-1]
                        
            NewRows = colstack([Ref0] + 
                  [deletecols(L[j][D[i]], keycols).repeat(Reps0) if i > 0 else 
                   deletecols(L[j][D[i]], keycols) for (i, j) in enumerate(I)  
                   if deletecols(L[j][D[i]], keycols) != None])
            for (i,j) in enumerate(I):
                L[j] = L[j][np.invert(D[i])]
            Result = rowstack([Result, NewRows], mode='nulls', 
                              nullvals=nullvals)

    return Result


def RenamingIsInCorrectFormat(renaming, L, Names=None):
    if isinstance(L, dict):
        Names = L.keys()
        LL = L.values()
    else:
        if Names == None:
            Names = range(len(L))
        else:
            assert len(Names) == len(L)
        LL = L

    return isinstance(renaming, dict) and \
           set(renaming.keys()) <= set(Names) and \
           all([isinstance(renaming[k],dict) and
                set(renaming[k].keys()) <= 
                set(LL[Names.index(k)].dtype.names) for k in renaming.keys()])


def DEFAULT_RENAMER(L, Names=None):
    """
    Renames overlapping column names of numpy ndarrays with structured dtypes

    Rename the columns by using a simple convention:

    *   If `L` is a list, it will append the number in the list to the key 
        associated with the array.

    *   If `L` is a dictionary, the algorithm will append the string 
        representation of the key associated with an array to the overlapping 
        columns from that array.

    Default renamer function used by :func:`tabular.spreadsheet.join`

    **Parameters**

            **L** :  list or dictionary

                    Numpy recarrays with columns to be renamed.

    **Returns**

            **D** :  dictionary of dictionaries

                    Dictionary mapping each input numpy recarray to a 
                    dictionary mapping each original column name to its new 
                    name following the convention above.

    """
    if isinstance(L,dict):
        Names = L.keys()
        LL = L.values()
    else:
        if Names == None:
            Names = range(len(L))
        else:
            assert len(Names) == len(L)
        LL = L

    commons = Commons([l.dtype.names for l in LL])

    D = {}
    for (i,l) in zip(Names, LL):
        d = {}
        for c in commons:
            if c in l.dtype.names:
                d[c] = c + '_' + str(i)
        if d:
            D[i] = d

    return D


def Commons(ListOfLists):
    commons = []
    for i in range(len(ListOfLists)):
        for j in range(i+1, len(ListOfLists)):
            commons.extend([l for l in ListOfLists[i] if l in ListOfLists[j]])
    return commons


def powerlist(S):
    if len(S) > 0:
        Sp = powerlist(S[:-1])
        return Sp + [x + [S[-1]] for x in Sp]
    else:
        return [[]]


def isunique(col):
    [D,s] = fast.recarrayuniqify(col)
    return len(D.nonzero()[0]) == len(col)