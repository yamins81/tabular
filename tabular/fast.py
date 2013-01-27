"""
Fast functions for manipulating and comparing numpy ndarrays (and recarrays), 
e.g. efficient NumPy algorithms for solving list-membership problems:

        arrayuniqify, recarrayuniqify,
        equalspairs, recarrayequalspairs,
        isin, recarrayisin,
        recarraydifference,
        arraymax, arraymin

"""

import numpy as np

__all__ = ['arrayuniqify', 'recarrayuniqify', 'equalspairs',       
           'recarrayequalspairs', 'isin', 'recarrayisin', 'recarraydifference', 
           'arraymax', 'arraymin']

def arrayuniqify(X, retainorder=False):
    """
    Very fast uniqify routine for numpy arrays.

    **Parameters**

            **X** :  numpy array

                    Determine the unique elements of this numpy array.

            **retainorder** :  Boolean, optional

                    Whether or not to return indices corresponding to unique 
                    values of `X` that also sort the values.  Default value is 
                    `False`, in which case `[D,s]` is returned.  This can be 
                    used to produce a uniqified version of `X` by simply 
                    taking::

                            X[s][D]

                    or::

                            X[s[D.nonzero()[0]]]

    **Returns**

            **D** :  numpy array

                    List of "first differences" in the sorted verion of `X`.  
                    Returned when `retainorder` is `False` (default).

            **s** :  numpy array

                    Permutation that will sort `X`.  Returned when 
                    `retainorder` is `False` (default).

            **ind** :  numpy array

                    List of indices that correspond to unique values of `X`, 
                    without sorting those values.  Returned when `retainorder` 
                    is `True`.

    **See Also:**

            :func:`tabular.fast.recarrayuniqify`

    """
    s = X.argsort()
    X = X[s]
    D = np.append([True],X[1:] != X[:-1])
    if retainorder:
        DD = np.append(D.nonzero()[0],len(X))
        ind = [min(s[x:DD[i+1]]) for (i,x) in enumerate(DD[:-1])]
        ind.sort()
        return ind
    else:
        return [D,s]


def recarrayuniqify(X, retainorder=False):
    """
    Very fast uniqify routine for numpy record arrays (or ndarrays with 
    structured dtype).

    Record array version of func:`tabular.fast.arrayuniqify`.

    **Parameters**

            **X** :  numpy recarray

                    Determine the unique elements of this numpy recarray.

            **retainorder** :  Boolean, optional

                    Whether or not to return indices corresponding to unique 
                    values of `X` that also sort the values.  Default value is 
                    `False`, in which case `[D,s]` is  returned.  This can be 
                    used to produce a uniqified  version of `X` by simply 
                    taking::

                            X[s][D]

                    or::

                            X[s[D.nonzero()[0]]]

    **Returns**

            **D** :  numpy recarray

                    List of "first differences" in the sorted verion of `X`.  
                    Returned when `retainorder` is `False` (default).

            **s** :  numpy array

                    Permutation that will sort `X`.  Returned when 
                    `retainorder` is `False` (default).

            **ind** :  numpy array

                    List of indices that correspond to unique values of `X`, 
                    without sorting those values.  Returned when `retainorder` 
                    is `True`.

    **See Also:**

            :func:`tabular.fast.arrayuniqify`

    """
    N = X.dtype.names
    s = X.argsort(order=N)
    s = s.view(np.ndarray)
    X = X[s]
    D = np.append([True],X[1:] != X[:-1])
    if retainorder:
        DD = np.append(D.nonzero()[0],len(X))
        ind = [min(s[x:DD[i+1]]) for (i,x) in enumerate(DD[:-1])]
        ind.sort()
        return ind
    else:
        return [D,s]


def equalspairs(X, Y):
    """
    Indices of elements in a sorted numpy array equal to those in another.

    Given numpy array `X` and sorted numpy array `Y`, determine the indices in 
    Y equal to indices in X.

    Returns `[A,B]` where `A` and `B` are numpy arrays of indices in `X` such 
    that::

            Y[A[i]:B[i]] = Y[Y == X[i]]`

    `A[i] = B[i] = 0` if `X[i]` is not in `Y`.

    **Parameters**

            **X** :  numpy array

                    Numpy array to compare to the sorted numpy array `Y`.

            **Y** :  numpy array

                    Sorted numpy array.  Determine the indices of elements of 
                    `Y` equal to those in numpy array `X`.

    **Returns**

            **A** :  numpy array

                    List of indices in `Y`, `len(A) = len(Y)`.

            **B** :  numpy array

                    List of indices in `Y`, `len(B) = len(Y)`.

    **See Also:**

            :func:`tabular.fast.recarrayequalspairs`

    """
    T = Y.copy()
    R = (T[1:] != T[:-1]).nonzero()[0]
    R = np.append(R,np.array([len(T)-1]))
    M = R[R.searchsorted(range(len(T)))]
    D = T.searchsorted(X)
    T = np.append(T,np.array([0]))
    M = np.append(M,np.array([0]))
    A = (T[D] == X) * D
    B = (T[D] == X) * (M[D] + 1)
    return [A,B]


def recarrayequalspairs(X,Y,weak=True):
    """
    Indices of elements in a sorted numpy recarray (or ndarray with 
    structured dtype) equal to those in another.

    Record array version of func:`tabular.fast.equalspairs`, but slightly 
    different because the concept of being sorted is less well-defined for a 
    record array.

    Given numpy recarray `X` and sorted numpy recarray `Y`, determine the 
    indices in Y equal to indices in X.

    Returns `[A,B,s]` where `s` is a permutation of `Y` such that for::

            Y = X[s]

    we have::

            Y[A[i]:B[i]] = Y[Y == X[i]]

    `A[i] = B[i] = 0` if `X[i]` is not in `Y`.

    **Parameters**

            **X** :  numpy recarray

                    Numpy recarray to compare to the sorted numpy recarray `Y`.

            **Y** :  numpy recarray

                    Sorted numpy recarray.  Determine the indices of elements 
                    of `Y` equal to those in numpy array `X`.

    **Returns**

            **A** :  numpy array

                    List of indices in `Y`, `len(A) = len(Y)`.

            **B** :  numpy array

                    List of indices in `Y`, `len(B) = len(Y)`.

            **s** :  numpy array

                    Permutation of `Y`.

    **See Also:**

            :func:`tabular.fast.recarrayequalspairs`

    """
    if (weak and set(X.dtype.names) != set(Y.dtype.names)) or \
       (not weak and X.dtype.names != Y.dtype.names):
        return [np.zeros((len(X),),int),np.zeros((len(X),),int),None]
    else:
        if X.dtype.names != Y.dtype.names:
            Y = np.rec.fromarrays([Y[a] for a in X.dtype.names], 
                                  names= X.dtype.names)
        NewX = np.array([str(l) for l in X])
        NewY = np.array([str(l) for l in Y])
        s = NewY.argsort()  ; NewY.sort()
        [A,B] = equalspairs(NewX,NewY)
        return [A,B,s]


def isin(X,Y):
    """
    Indices of elements in a numpy array that appear in another.

    Fast routine for determining indices of elements in numpy array `X` that 
    appear in numpy array `Y`, returning a boolean array `Z` such that::

            Z[i] = X[i] in Y

    **Parameters**

            **X** :  numpy array

                    Numpy array to comapare to numpy array `Y`.  For each 
                    element of `X`, ask if it is in `Y`.

            **Y** :  numpy array

                    Numpy array to which numpy array `X` is compared.  For each 
                    element of `X`, ask if it is in `Y`.

    **Returns**

            **b** :  numpy array (bool)

                    Boolean numpy array, `len(b) = len(X)`.

    **See Also:**

            :func:`tabular.fast.recarrayisin`, 
            :func:`tabular.fast.arraydifference`

    """
    if len(Y) > 0:
        T = Y.copy()
        T.sort()
        D = T.searchsorted(X)
        T = np.append(T,np.array([0]))
        W = (T[D] == X)
        if isinstance(W,bool):
            return np.zeros((len(X),),bool)
        else:
            return (T[D] == X)
    else:
        return np.zeros((len(X),),bool)



def recarrayisin(X,Y,weak=True):
    """
    Indices of elements in a numpy record array (or ndarray with structured 
    dtype) that appear in another.

    Fast routine for determining indices of elements in numpy record array `X` 
    that appear in numpy record array `Y`, returning a boolean array `Z` such 
    that::

            Z[i] = X[i] in Y

    Record array version of func:`tabular.fast.isin`.

    **Parameters**

            **X** :  numpy recarray

                    Numpy recarray to comapare to numpy recarray `Y`.  For each 
                    element of `X`, ask if it is in `Y`.

            **Y** :  numpy recarray

                    Numpy recarray to which numpy recarray `X` is compared.  
                    For each element of `X`, ask if it is in `Y`.

    **Returns**

            **b** :  numpy array (bool)

                    Boolean numpy array, `len(b) = len(X)`.

    **See Also:**

            :func:`tabular.fast.isin`, :func:`tabular.fast.recarraydifference`

    """
    if (weak and set(X.dtype.names) != set(Y.dtype.names)) or \
       (not weak and X.dtype.names != Y.dtype.names):
        return np.zeros((len(X),),bool)
    else:
        if X.dtype.names != Y.dtype.names:
            Y = np.rec.fromarrays([Y[a] for a in X.dtype.names], 
                                  names=X.dtype.names)
        NewX = np.array([str(l) for l in X])
        NewY = np.array([str(l) for l in Y])
        NewY.sort()
        return isin(NewX,NewY)


def arraydifference(X,Y):
    """
    Elements of a numpy array that do not appear in another.

    Fast routine for determining which elements in numpy array `X`
    do not appear in numpy array `Y`.

    **Parameters**

            **X** :  numpy array

                    Numpy array to comapare to numpy array `Y`.
                    Return subset of `X` corresponding to elements not in `Y`.

            **Y** :  numpy array

                    Numpy array to which numpy array `X` is compared.
                    Return subset of `X` corresponding to elements not in `Y`.

    **Returns**

            **Z** :  numpy array

                    Subset of `X` corresponding to elements not in `Y`.

    **See Also:**

            :func:`tabular.fast.recarraydifference`, :func:`tabular.fast.isin`

    """
    if len(Y) > 0:
        Z = isin(X,Y)
        return X[np.invert(Z)]
    else:
        return X


def recarraydifference(X,Y):
    """
    Records of a numpy recarray (or ndarray with structured dtype)
    that do not appear in another.

    Fast routine for determining which records in numpy array `X`
    do not appear in numpy recarray `Y`.

    Record array version of func:`tabular.fast.arraydifference`.

    **Parameters**

            **X** :  numpy recarray

                    Numpy recarray to comapare to numpy recarray `Y`.
                    Return subset of `X` corresponding to elements not in `Y`.

            **Y** :  numpy recarray

                    Numpy recarray to which numpy recarray `X` is compared.
                    Return subset of `X` corresponding to elements not in `Y`.

    **Returns**

            **Z** :  numpy recarray

                    Subset of `X` corresponding to elements not in `Y`.

    **See Also:**

            :func:`tabular.fast.arraydifference`, :func:`tabular.fast.recarrayisin`

    """
    if len(Y) > 0:
        Z = recarrayisin(X,Y)
        return X[np.invert(Z)]
    else:
        return X


def arraymax(X,Y):
    """
    Fast "vectorized" max function for element-wise comparison of two numpy arrays.

    For two numpy arrays `X` and `Y` of equal length,
    return numpy array `Z` such that::

            Z[i] = max(X[i],Y[i])

    **Parameters**

            **X** :  numpy array

                    Numpy array; `len(X) = len(Y)`.

            **Y** :  numpy array

                    Numpy array; `len(Y) = len(X)`.

    **Returns**

            **Z** :  numpy array

                    Numpy array such that `Z[i] = max(X[i],Y[i])`.

    **See Also**

            :func:`tabular.fast.arraymin`

    """
    Z = np.zeros((len(X),), int)
    A = X <= Y
    B = Y < X
    Z[A] = Y[A]
    Z[B] = X[B]
    return Z


def arraymin(X,Y):
    """
    Fast "vectorized" min function for element-wise comparison of two 
    numpy arrays.

    For two numpy arrays `X` and `Y` of equal length,
    return numpy array `Z` such that::

            Z[i] = min(X[i],Y[i])

    **Parameters**

            **X** :  numpy array

                    Numpy array; `len(X) = len(Y)`.

            **Y** :  numpy array

                    Numpy array; `len(Y) = len(X)`.

    **Returns**

            **Z** :  numpy array

                    Numpy array such that `Z[i] = max(X[i],Y[i])`.

    **See Also**

            :func:`tabular.fast.arraymax`

    """
    Z = np.zeros((len(X),), int)
    A = X <= Y
    B = Y < X
    Z[A] = X[A]
    Z[B] = Y[B]
    return Z
