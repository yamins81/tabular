"""
Run from tabular/

"""

import unittest
import cPickle
import os
import shutil
import types
import numpy as np
import tabular as tb
import tabular.utils as utils
import tabular.spreadsheet as spreadsheet

def delete(ToDelete):
    '''
    Unified "strong" version of delete.
    
    Uses os.remove for a file and shutil.rmtree for a directory tree.
    '''
    if os.path.isfile(ToDelete):
        os.remove(ToDelete)
    elif os.path.isdir(ToDelete):
        shutil.rmtree(ToDelete)

def makedir(DirName):
    '''
    "Strong" directory maker:  if DirName already exists, delete it first.
    '''
    if os.path.exists(DirName):
        delete(DirName)
    os.mkdir(DirName)

TestDataDir = 'tests/tabularTestData/'
makedir(TestDataDir)

class TesterCore(unittest.TestCase):
    def setUp(self):
        self.D = tb.tabarray(array = np.random.rand(10**3 + 50, 10**2))
        self.Root = 'big'

    def assert_io(self, expr, fname):
        if expr:
            delete(fname)
            self.assert_(expr)
        else:
            self.assert_(expr)

    def test_empty(self):
        D = tb.tabarray(dtype=self.D.dtype, coloring=self.D.coloring)
        self.assert_(self.D.dtype==D.dtype)

    def test_save_load_TSV(self):
        fname = TestDataDir + self.Root + '.tsv'
        print fname
        self.D.saveSV(fname, metadata=['names', 'formats', 'types', 'coloring', 'dialect'])
        D = tb.tabarray(SVfile = fname)
        self.assert_io(eq(self.D, D), fname)

    def test_save_load_CSV(self):
        fname = TestDataDir + self.Root + '.csv'
        self.D.saveSV(fname, metadata=['names', 'formats', 'types', 'coloring', 'dialect'])
        D = tb.tabarray(SVfile = fname)
        self.assert_io(eq(self.D, D), fname)

    def test_save_load_binary(self):
        fname = TestDataDir + self.Root + '.npz'
        self.D.savebinary(fname)
        D = tb.tabarray(binary = fname)
        self.assert_io(eq(self.D, D), fname)

    def test_save_load_binary_data_only(self):
        fname = TestDataDir + self.Root + '.npy'
        self.D.savebinary(fname)
        D = tb.tabarray(binary = fname)
        self.assert_io(all(self.D == D), fname)

    def test_addrecords_tuple(self):
        D = self.D[:-1].copy()
        x = self.D[-1].tolist()
        D1 = D.addrecords(x)
        self.assert_(isinstance(x, tuple) & eq(D1,self.D))

    def test_addrecords_void(self):
        D = self.D[:-1].copy()
        x = np.array([self.D[-1]], dtype=self.D.dtype.descr)[0]
        D1 = D.addrecords(x)
        self.assert_(isinstance(x, np.void) & eq(self.D, D1))

    def test_addrecords_record(self):
        D = self.D[:-1].copy()
        x = self.D[-1]
        D1 = D.addrecords(x)
        self.assert_(eq(self.D, D1))

    def test_addrecords_tuples(self):
        ind = len(self.D)/2
        D = self.D[:ind].copy()
        x = self.D[ind:].tolist()
        D1 = D.addrecords(x)
        self.assert_(isinstance(x[0], tuple) & eq(self.D, D1))

    def test_addrecords_voids(self):
        ind = len(self.D)/2
        D = self.D[:ind].copy()
        x = np.array([rec for rec in self.D[ind:].tolist()], dtype=self.D.dtype.descr)
        x = [v for v in x]
        D1 = D.addrecords(x)
        self.assert_(isinstance(x[0], np.void) & eq(self.D, D1))

    def test_addrecords_records(self):
        ind = len(self.D)/2
        D = self.D[:ind].copy()
        x = self.D[ind:]
        D1 = D.addrecords(x)
        self.assert_(eq(self.D, D1))

    def test_rowstack(self):
        ind = len(self.D)/2
        self.assert_(eq(self.D, (self.D[:ind]).rowstack(self.D[ind:])))

    def test_colstack(self):
        names = list(self.D.dtype.names)
        ind = len(names)/2
        self.assert_(all(self.D == (self.D[names[:ind]]).colstack(self.D[names[ind:]])))

    def test_equals(self):
        D = self.D
        self.assert_(eq(self.D, D) and self.D is D)

    def test_equals_copy(self):
        D = self.D.copy()
        self.assert_(eq(self.D, D) and not self.D is D)

def test_colstack_renaming():
    X = tb.tabarray(columns= [['a','b'],[1,2]],names = ['A','B'],coloring={'Categories':['A']})
    Y = tb.tabarray(columns= [['c','d'],[3,4]],names = ['A','B'],coloring={'Categories':['A']})
    Z = X.colstack(Y,mode='rename')
    Z2 = tb.tabarray(columns = [['a','b'],[1,2],['c','d'],[3,4]],names = ['A_0','B_0','A_1','B_1'],coloring={'Categories':['A_0','A_1']})
    assert eq(Z,Z2)


class TestBasic(TesterCore):

    def setUp(self):
        self.D = tb.tabarray(
                 array=[(2, 'a', 2, 'cc', 3.0), (2, 'b', 5, 'dcc', 2.0), 
                        (7, 'e', 2, 'cjc', 8.0), (2, 'e', 2, 'ccj', 3.0)], 
                 names=['a', 'c', 'b', 'd', 'e'], formats='i4,|S1,i4,|S3,f8', 
                 coloring={'moo': ['a', 'b'], 'boo': ['a', 'd', 'e']})
        self.Root = 'basic'

    def tearDown(self):
        self.D = None
        self.Root = None

    def test_getitem_list_order(self):  
        assert self.D[['a', 'b']].dtype.names == ('a', 'b')
        assert self.D[['b', 'a']].dtype.names == ('b', 'a')

    def test_getitem_color(self):
        self.assert_(eq(self.D['moo'], self.D[['a', 'b']]))

    def test_getitem_color_threshold(self):
        self.assertEqual(self.D[['a', 'b']].coloring, {'moo':['a','b'],'boo': ['a']})

    def test_getitem_list_colors(self):
        self.assert_(eq(self.D[['a', 'boo']], self.D['boo']))

    def test_replace_int(self):
        D = self.D.copy()
        D.replace(2, 100, cols=['a', 'b', 'e'])
        x = self.D[['a', 'b', 'e']].extract()
        x[x == 2] = 100
        self.assert_((D[['a', 'b', 'e']].extract() == x).all())

    def test_replace_float(self):
        D = self.D.copy()
        D.replace(2.0, 100.0, cols=['a', 'b', 'e'])
        x = self.D[['a', 'b', 'e']].extract()
        x[x == 2.0] = 100.0
        self.assert_((D[['a', 'b', 'e']].extract() == x).all())

    def test_replace_str(self):
        D = self.D.copy()
        D.replace('cc', 'bb', cols=['d'])
        x = self.D.copy()['d']
        x[x == 'cc'] = 'bb'
        self.assert_(all(D['d'] == x))

    def test_replace_str_notstrict(self):
        D = self.D.copy()
        D.replace('cc', 'bb', cols=['d'], strict=False)
        x = self.D.copy()['d']
        x = [row.replace('cc', 'bb') for row in x]
        self.assert_(all(D['d'] == x))

    def test_replace_rows(self):
        D = self.D.copy()
        D.replace('cc', 'bb', cols=['d'], rows=D['a']==2)
        x = self.D.copy()['d']
        x[(x == 'cc') & (D['a'] == 2)] = 'bb'
        self.assert_(all(D['d'] == x))

    def test_toload_tsv(self):
        toload = ['c', 'boo']
        fname = TestDataDir + self.Root + '5.tsv'
        self.D.saveSV(fname, metadata=['names', 'formats', 'types', 'coloring', 'dialect'])
        D = tb.tabarray(SVfile=fname, usecols=toload)
        assert set(D.dtype.names) == set(['c'] + D.coloring['boo'])
        self.assert_io(eq(self.D[toload], D[toload]), fname)

    def test_toload_redundant_tsv(self):
        toload = ['a', 'boo']
        fname = TestDataDir + self.Root + '6.tsv'
        self.D.saveSV(fname, metadata=['names', 'formats', 'types', 'coloring', 'dialect'])
        D = tb.tabarray(SVfile=fname, usecols=toload)
        assert set(D.dtype.names) == set(D.coloring['boo'])
        self.assert_io(eq(self.D[toload], D[toload]), fname)

    def test_aggregate_AggFunc(self):
        AggFunc=np.mean
        [D1,s] = self.D[['a', 'b', 'e']].aggregate(
                                     On=['e'], AggFunc=AggFunc,returnsort=True)
        e = utils.uniqify(self.D['e'][s])
        a = []
        b = []
        for i in e:
            boolvec = self.D['e'][s] == i
            a += [AggFunc(self.D['a'][s][boolvec])]
            b += [AggFunc(self.D['b'][s][boolvec])]
        D2 = tb.tabarray(columns=[e,a,b], names=['e','a','b'], coloring=D1.coloring)
        self.assert_(eq(D1,D2))

    def test_aggregate1(self):
        AggFuncDict = {'d': ','.join}
        [D1,s] = self.D[['a', 'b', 'd']].aggregate(
                             On=['a'], AggFuncDict=AggFuncDict,returnsort=True)
        a = utils.uniqify(self.D['a'][s])
        AggFuncDict.update({'b': sum})
        b = []
        d = []
        for i in a:
            boolvec = self.D['a'][s] == i
            b += [AggFuncDict['b'](self.D['b'][s][boolvec])]
            d += [AggFuncDict['d'](self.D['d'][s][boolvec])]
        D2 = tb.tabarray(columns=[a, b, d], names=['a', 'b', 'd'], coloring=D1.coloring)
        self.assert_(eq(D1, D2))

    def test_aggregate2(self):
        AggFuncDict = {'c': '+'.join, 'd': ','.join}
        [D1,s] = self.D[['a', 'c', 'b', 'd']].aggregate(
                        On=['a', 'b'], AggFuncDict=AggFuncDict,returnsort=True)
        ab = utils.uniqify(zip(self.D['a'][s], self.D['b'][s]))
        c = []
        d = []
        for i in ab:
            boolvec = np.array([tuple(self.D[['a', 'b']][s][ind])==i 
                                for ind in range(len(self.D))])
            c += [AggFuncDict['c'](self.D['c'][s][boolvec])]
            d += [AggFuncDict['d'](self.D['d'][s][boolvec])]
        D2 = tb.tabarray(
             columns=[[x[0] for x in ab],[x[1] for x in ab], c,d], 
             names=['a', 'b','c', 'd'], coloring=D1.coloring)
        self.assert_(eq(D1, D2))

    def test_aggregate_in(self):
        AggFuncDict = {'c': '+'.join, 'd': ','.join}
        D = self.D.aggregate(On=['a','b'], AggFuncDict=AggFuncDict)
        D1 = self.D.aggregate_in(On=['a','b'], AggFuncDict=AggFuncDict, 
                 interspersed=False).deletecols(['__aggregates__','__color__'])
        D2 = self.D.rowstack(D)
        self.assert_(all(D1==D2))

class TestAddCols(unittest.TestCase):
    def setUp(self):
        V1 = ['North','South','East','West']
        V2 = ['Service','Manufacturing','Education','Healthcare']
        Recs = [(a, b, np.random.rand() * 100, np.random.randint(100000), 
                 np.random.rand(),'Yes' if np.random.rand() < .5 else 'No') 
                for a in V1 for b in V2]
        self.Y = tb.tabarray(records=Recs, names=['Region', 'Sector', 'Amount', 
                                     'Population', 'Importance', 'Modernized'])
        self.n1 = 'Importance'
        self.n2 = 'Modernized'
        oklist = [o for o in self.Y.dtype.names 
                         if o not in [self.n1, self.n2]]
        self.X = self.Y[oklist]

    def test_1(self):
        Y = self.X.addcols(self.Y[self.n1], names=self.n1)
        self.assert_(eq(Y,self.Y[[o for o in self.Y.dtype.names 
                                  if o != self.n2]]))

    def test_2(self):
        Y = self.X.addcols(list(self.Y[self.n1]), names=[self.n1])
        self.assert_(eq(Y,self.Y[[o for o in self.Y.dtype.names 
                                  if o != self.n2]]))

    def test_3(self):
        z = np.rec.fromarrays([self.Y[self.n1]], names=[self.n1])
        Y = self.X.addcols(z)
        self.assert_(eq(Y,self.Y[[o for o in self.Y.dtype.names 
                                  if o != self.n2]]))

    def test_4(self):
        Y = self.X.addcols([self.Y[self.n1], self.Y[self.n2]], 
                           names=[self.n1, self.n2])
        self.assert_(eq(Y,self.Y))

    def test_5(self):
        Y = self.X.addcols([self.Y[self.n1], list(self.Y[self.n2])], 
                           names=self.n1 + ',' + self.n2)
        self.assert_(eq(Y,self.Y))

    def test_6(self):
        Y = self.X.addcols([list(self.Y[self.n1]), list(self.Y[self.n2])], 
                           names=(self.n1 + ', ' + self.n2))
        self.assert_(eq(Y,self.Y))

    def test_7(self):
        Y = self.X.addcols(self.Y[[self.n1,self.n2]])
        self.assert_(eq(Y,self.Y))


class TestJoin(unittest.TestCase):
    def setUp(self):
        V1 = ['North', 'South', 'East', 'West']
        V2 = ['Service', 'Manufacturing', 'Education', 'Healthcare']
        Recs = [(a, b, np.random.rand() * 100, np.random.randint(100000), 
                 np.random.rand(), 'Yes' if np.random.rand() < .5 else 'No') 
                for a in V1 for b in V2]
        self.X = tb.tabarray(records=Recs, names=['Region', 'Sector', 'Amount', 
                                     'Population', 'Importance', 'Modernized'])
        self.keycols = ['Region', 'Sector']
        self.others = [['Amount', 'Population'], ['Importance', 'Modernized']]

    def test_strictjoin(self):
        ToMerge = [self.X[self.keycols + n] for n in self.others]
        Y = spreadsheet.strictjoin(ToMerge,self.keycols)
        Z = self.X.copy()
        Z.sort(order  = self.keycols)
        Y.sort(order = self.keycols)
        self.assert_((Z == Y).all())

    def test_strictjoin2(self):
        ToMerge = [self.X[self.keycols + [x]] for x in self.X.dtype.names 
                   if x not in self.keycols]
        Y = spreadsheet.strictjoin(ToMerge, self.keycols)
        Z = self.X.copy()
        Z.sort(order=self.keycols)
        Y.sort(order=self.keycols)
        self.assert_((Z == Y).all())

    def test_strictjoin3(self):
        X = self.X
        keycols = self.keycols
        others=self.others
        X1 = X[:(3 * len(X) / 4)][keycols + others[0]]
        X2 = X[(len(X) / 4):][keycols + others[1]]
        Y = spreadsheet.strictjoin([X1, X2], self.keycols)
        Y.sort(order=keycols)

        nvf = utils.DEFAULT_NULLVALUEFORMAT
        nvf1 = nvf(X[others[1][0]].dtype.descr[0][1])
        nvf2 = nvf(X[others[1][1]].dtype.descr[0][1])
        nvf3 = nvf(X[others[0][0]].dtype.descr[0][1])
        nvf4 = nvf(X[others[0][1]].dtype.descr[0][1])

        Recs = ([(a, b, c, d, nvf1, nvf2) for (a, b, c, d, e, f) 
                                         in X[:(len(X) / 4)]] + 
                [(a, b, c, d, e, f) for (a, b, c, d, e, f) 
                                   in X[(len(X) / 4):(3 * len(X) / 4)]] + 
                [(a, b, nvf3, nvf4, e, f) for (a, b, c, d, e, f) 
                                         in X[(3 * len(X) / 4):]])
        Z = tb.tabarray(records=Recs, names=X.dtype.names)
        Z.sort(order=self.keycols)

        self.assert_((Y == Z).all())

    def test_strictjoin4(self):
        ToMerge = dict([('d' + str(i) , self.X[self.keycols + n]) 
                        for (i, n) in enumerate(self.others)])
        Y = spreadsheet.strictjoin(ToMerge, self.keycols)
        Y.sort(order=self.keycols)
        Z = self.X.copy()
        Z.sort(order=self.keycols)
        self.assert_((Z == Y).all())


    def test_join(self):
        ToMerge = [self.X[self.keycols + n] for n in self.others]
        Y = spreadsheet.join(ToMerge)
        Y.sort(order = self.keycols)
        Z = self.X.copy()
        Z.sort(order  = self.keycols)
        self.assert_((Z == Y).all())

    def test_join2(self):
        Y1 = self.X[['Region', 'Sector', 'Amount']].copy()
        Y2 = self.X[['Region', 'Sector', 'Modernized']].copy()
        Y1.renamecol('Amount', 'Modernized')
        Z = spreadsheet.join([Y1, Y2], ['Region', 'Sector'])

        Z1 = self.X[['Region', 'Sector', 'Amount', 'Modernized']]
        Z1.sort()
        Z1.renamecol('Amount', 'Modernized_0')
        Z1.renamecol('Modernized', 'Modernized_1')

        self.assert_((Z1 == Z).all())

    def test_join3(self):
        Recs1 = [('North', 'Service', 80.818237828506838),
                 ('North', 'Manufacturing', 67.065114829789664), 
                 ('North', 'Education', 31.043641435185123), 
                 ('North', 'Healthcare', 14.196823211749276), 
                 ('South', 'Service',2.3583798234914521)]
        Recs2 = [('North', 'Service', 33.069022471086903), 
                 ('North', 'Manufacturing', 63.155520758932305), 
                 ('North', 'Education', 70.80529023970098), 
                 ('North', 'Healthcare', 40.301231798570171), 
                 ('South', 'Service', 13.095729670745381)]
        X1 = tb.tabarray(records=Recs1, names=['Region', 'Sector', 'Amount'])
        X2 = tb.tabarray(records=Recs2, names=['Region', 'Sector', 'Amount'])
        Z = spreadsheet.join([X1, X2], keycols=['Region', 'Sector'], 
                             Names=['US', 'China'])

        Recs = [(a, b, c, d) for ((a,b,c),(x1,x2,d)) in zip(Recs1,Recs2)]
        X = tb.tabarray(records=Recs, 
                       names=['Region', 'Sector', 'Amount_US', 'Amount_China'])

        X.sort(order=['Region', 'Sector'])
        Z.sort(order=['Region', 'Sector'])
        assert (X == Z).all()

    def test_joinmethod(self):
        X = self.X.copy()
        keycols = self.keycols
        others = self.others
        Y = X[keycols + others[0]]
        Z = Y.join([X[keycols + n] for n in others[1:]])
        X.sort(order=keycols)
        Z.sort(order=keycols)
        self.assert_(eq(X, Z))

    def test_joinmethod2(self):
        X = self.X.copy()
        keycols = self.keycols
        others = self.others
        X.coloring['Numerical'] = ['Amount', 'Population', 'Importance']

        Y1 = X[['Region', 'Sector', 'Amount']].copy()
        Y2 = X[['Region', 'Sector', 'Modernized']].copy()
        Y1.renamecol('Amount', 'Modernized')
        Z = Y1.join(Y2, keycols=['Region', 'Sector'])

        Z1 = X[['Region', 'Sector', 'Amount', 'Modernized']]
        Z1.sort()
        Z1.renamecol('Amount', 'Modernized_0')
        Z1.renamecol('Modernized', 'Modernized_1')

        self.assert_(eq(Z, Z1))

def assert_bio(expr, fname):
    if expr:
        delete(fname)
        assert(expr)
    else:
        assert(expr)

def test_bionumbers():
    X = tb.tabarray(SVfile = 'tests/bionumbers.txt') 
    fname = TestDataDir + 'bionumbers.txt'
    X.saveSV(fname, quotechar="'")
    Y = tb.tabarray(SVfile = TestDataDir + 'bionumbers.txt',quotechar="'")
    names = ('ID', 'Property', 'Organism', 'Value', 'Units', 'Range', 
              'NumericalValue', 'Version')               
    assert_bio(X.dtype.names == names and len(X) == 4615 and eq(X,Y), fname)
     
class TestLoadSaveSV(unittest.TestCase):        # test non-default use cases

    def assert_io(self, expr, fname):
        if expr:
            delete(fname)
            self.assert_(expr)
        else:
            self.assert_(expr)

    def setUp(self):
        V1 = ['North', 'South', 'East', 'West']
        V2 = ['Service', 'Manufacturing', 'Education', 'Healthcare']
        Recs = [(a, b, np.random.rand() * 100, np.random.randint(100)) 
                for a in V1 for b in V2]
        self.X = tb.tabarray(records=Recs,         
                           names=['Region', 'Sector', 'Amount', 'Population'], 
                           coloring={'zoo': ['Region','Sector'], 
                                     'york': ['Population','Sector','Region']})

    def test_load_save_CSV_infer(self):
        fname = TestDataDir + 'test.csv'
        self.X.saveSV(fname)
        X2 = tb.tabarray(SVfile=fname)  # normal scenario: names, no comments
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_load_save_TSV_infer(self):
        fname = TestDataDir + 'test.tsv'
        self.X.saveSV(fname)
        X2 = tb.tabarray(SVfile=fname)
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_load_save_CSV_infer1(self):
        fname = TestDataDir + 'test1.csv'
        self.X.saveSV(fname, metadata=True)  # default metadata settings
        X2 = tb.tabarray(SVfile=fname)
        self.assert_io(eq(self.X, X2), fname)

    def test_load_save_TSV_infer1(self):
        fname = TestDataDir + 'test1.tsv'
        self.X.saveSV(fname, metadata=True)
        X2 = tb.tabarray(SVfile=fname)
        self.assert_io(eq(self.X, X2), fname)

    def test_load_save_CSV_infer2(self):
        fname = TestDataDir + 'test2.csv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'])
        X2 = tb.tabarray(SVfile=fname, 
                         metametadata={'coloring': 0, 'names': 1})
        self.assert_io(eq(self.X, X2), fname)

    def test_load_save_TSV_infer2(self):
        fname = TestDataDir + 'test2.tsv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'])
        X2 = tb.tabarray(SVfile=fname, 
                         metametadata={'coloring': 0, 'names': 1})
        self.assert_io(eq(self.X, X2), fname)

    def test_load_save_CSV_skiprows(self):
        fname = TestDataDir + 'test3.csv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'])
        X2 = tb.tabarray(SVfile=fname, skiprows=1)
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_load_save_TSV_skiprows(self):
        fname = TestDataDir + 'test3.tsv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'])
        X2 = tb.tabarray(SVfile=fname, skiprows=1)
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_load_save_CSV_nocomments(self):
        fname = TestDataDir + 'test4.csv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'], comments='')
        X2 = tb.tabarray(SVfile=fname, comments='', headerlines=2)
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_load_save_TSV_nocomments(self):
        fname = TestDataDir + 'test4.tsv'
        self.X.saveSV(fname, printmetadict=False, 
                      metadata=['coloring', 'names'], comments='')
        X2 = tb.tabarray(SVfile=fname, headerlines=2)
        Z = self.X.copy()
        Z.coloring = {}
        self.assert_io(eq(Z, X2), fname)

    def test_linefixer(self):
        fname = TestDataDir + 'linefixer.txt'
        X1 = self.X.copy()
        X1.coloring = {}
        X1.saveSV(fname, delimiter='@')   
        X2 = tb.tabarray(SVfile=fname, 
                         linefixer=(lambda x: x.replace('@','\t')))

        self.assert_io(eq(X1, X2), fname)                      

    def test_valuefixer(self):
        fname = TestDataDir + 'valuefixer.txt'
        X = self.X
        X.coloring = {}
        X1 = X.copy()
        X1['Population'] = X1['Population'] + 1
        X1.saveSV(fname)
        X2 = tb.tabarray(SVfile=fname, 
                    valuefixer=(lambda x: str(int(x)-1) if x.isdigit() else x))           
        print X1, X2
        self.assert_io(eq(X, X2), fname)
        
    def test_missingvals(self):
        fname = TestDataDir + 'missingvals.csv'
        F = open(fname,'w')
        F.write('Name,Age,Gender\nDaniel,12,M\nElaine,N/A,F\nFarish,46,')
        F.close()
        X = tb.tabarray(SVfile=fname,fillingvalues={'Gender':('','N/A'),'Age':('N/A',-1)})
        X2 = tb.tabarray(records=[('Daniel', 12, 'M'), ('Elaine', -1, 'F'), ('Farish', 46, 'N/A')],names=['Name','Age','Gender'])
        self.assert_io(eq(X, X2), fname)

    def test_missingvals2(self):
        fname = TestDataDir + 'missingvals2.csv'
        F = open(fname,'w')
        F.write('Name,Age,Gender\nDaniel,12,M\nElaine,N/A,F\nFarish,46,')
        F.close()
        X = tb.tabarray(SVfile=fname,fillingvalues={2:('','N/A'),'Age':('N/A',-1)})
        X2 = tb.tabarray(records=[('Daniel', 12, 'M'), ('Elaine', -1, 'F'), ('Farish', 46, 'N/A')],names=['Name','Age','Gender'])
        self.assert_io(eq(X, X2), fname)

    def test_missingvals3(self):
        fname = TestDataDir + 'missingvals3.csv'
        F = open(fname,'w')
        F.write('Name,Age,Gender\nDaniel,12,M\nElaine,N/A,F\nFarish,46,')
        F.close()
        X = tb.tabarray(SVfile=fname,fillingvalues=(('',''),('N/A',-1),('','N/A')))
        X2 = tb.tabarray(records=[('Daniel', 12, 'M'), ('Elaine', -1, 'F'), ('Farish', 46, 'N/A')],names=['Name','Age','Gender'])
        self.assert_io(eq(X, X2), fname)        
                
    def test_missingvals4(self):
        fname = TestDataDir + 'missingvals4.csv'
        F = open(fname,'w')
        F.write('Name,Age,Gender\nDaniel,12,M\nElaine,'',F\nFarish,46,')
        F.close()
        X = tb.tabarray(SVfile=fname)
        X2 = tb.tabarray(records=[('Daniel', 12, 'M'), ('Elaine', np.nan, 'F'), ('Farish', 46, '')],names=['Name','Age','Gender'])
        self.assert_io(eq(X, X2), fname)                        

    def test_missingvals5(self):
        fname = TestDataDir + 'missingvals5.csv'
        F = open(fname,'w')
        F.write('Name,Age,Gender\nDaniel,12,M\nElaine,N/A,F\nFarish,46,')
        F.close()
        X = tb.tabarray(SVfile=fname,missingvalues={'Age':'N/A'})
        X2 = tb.tabarray(records=[('Daniel', 12, 'M'), ('Elaine', np.nan, 'F'), ('Farish', 46, '')],names=['Name','Age','Gender'])
        print X, X2
        self.assert_io(eq(X, X2), fname)
                    
class TestLoadSaveSVTutorial(unittest.TestCase):

    def assert_io(self, expr, fname):
        if expr:
            delete(fname)
            self.assert_(expr)
        else:
            self.assert_(expr)

    def setUp(self):
        names = ['name', 'ID', 'color', 'size', 'June', 'July']
        data = [('bork', 1212, 'blue', 'big', 45.32, 46.07), 
                ('mork', 4660, 'green', 'small', 32.18, 32.75), 
                ('stork', 2219, 'red', 'huge', 60.93, 61.82), 
                ('lork', 4488, 'purple', 'tiny', 0.44, 0.38)]
        self.x = tb.tabarray(records=data, names=names)

    def test_tsv(self):
        fname = TestDataDir + 'example_copy.tsv'
        self.x.saveSV(fname)
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_csv(self):
        fname = TestDataDir + 'example.csv'
        self.x.saveSV(fname)
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_tsvcsv(self):
        fname = TestDataDir + 'tab.csv'
        self.x.saveSV(fname, delimiter='\t')
        x = tb.tabarray(SVfile=fname, delimiter='\t')
        self.assert_io(eq(x, self.x), fname)

    def test_nonames(self):
        fname = TestDataDir + 'nonames.tsv'
        self.x.saveSV(fname, metadata=False)
        x = tb.tabarray(SVfile=fname, namesinheader=False)
        y = tb.tabarray(records=x.tolist())
        self.assert_io(eq(x, y), fname)

    def test_hash(self):
        fname = TestDataDir + 'hash.tsv'
        self.x.saveSV(fname, comments='#')
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_comments(self):
        fname = TestDataDir + 'comments.tsv'
        self.x.saveSV(fname, comments='@')
        x = tb.tabarray(SVfile=fname, comments='@')
        self.assert_io(eq(x, self.x), fname)

    def test_verbose(self):
        fname = TestDataDir + 'verbose.tsv'
        self.x.saveSV(fname, comments='#')
        f = open(fname, 'r').read()
        g = open(fname, 'w')
        verbose = '\n'.join(['#this is my file', '#these are my verbose notes', 
                             '#blah blah blah'])
        g.write(verbose + '\n' + f)
        g.close()
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_nohash(self):
        fname = TestDataDir + 'nohash.tsv'
        self.x.saveSV(fname, comments='')
        f = open(fname, 'r').read()
        g = open(fname, 'w')
        g.write('this is my file\n' + f)
        g.close()
        x = tb.tabarray(SVfile=fname, headerlines=2)
        self.assert_io(eq(x, self.x), fname)

    def test_meta_names(self):
        fname = TestDataDir + 'meta_names.tsv'
        self.x.saveSV(fname, metadata=['names'])
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_meta_nohash(self):
        fname = TestDataDir + 'meta_nohash.tsv'
        self.x.saveSV(fname)
        f = open(fname, 'r').read()
        g = open(fname, 'w')
        header = '\n'.join(["metametadata={'names': 3}", 
                            'these are my verbose notes', 'blah blah blah'])
        g.write(header + '\n' + f)
        g.close()
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_meta_random(self):
        fname = TestDataDir + 'meta_random.tsv'
        self.x.saveSV(fname)
        f = open(fname, 'r').read().split('\n')
        g = open(fname, 'w')
        header = '\n'.join(['#this is my file', "#metametadata={'names': 2}", 
                         '#'+ f[0], '#not sure why there are more notes here'])
        g.write(header + '\n' + '\n'.join(f[1:]))
        g.close()
        x = tb.tabarray(SVfile=fname)
        print x, x.dtype.names,x.coloring
        print self.x, self.x.dtype.names, self.x.coloring
        self.assert_io(eq(x, self.x), fname)

    def test_meta_types(self):
        fname = TestDataDir + 'meta_types.tsv'
        self.x.saveSV(fname, metadata=True)
        x = tb.tabarray(SVfile=fname)
        self.assert_io(eq(x, self.x), fname)

    def test_meta_types1(self):
        fname = TestDataDir + 'meta_types1.tsv'
        self.x.saveSV(fname, metadata=True, printmetadict=False, comments='')
        x = tb.tabarray(SVfile=fname,  metametadata={'names': 2, 'types': 1})
        self.assert_io(eq(x, self.x), fname)

    def test_skiprows(self):
        fname = TestDataDir + 'skiprows.tsv'
        self.x.saveSV(fname, metadata=True)
        x = tb.tabarray(SVfile=fname, skiprows=3)
        self.assert_io(eq(x, self.x), fname)

    def test_usecols(self):
        fname = TestDataDir + 'usecols.tsv'
        self.x.saveSV(fname)
        x = tb.tabarray(SVfile=fname, usecols=[0,-1])
        names=[self.x.dtype.names[i] for i in [0,-1]]
        print x,x.dtype.names
        print self.x[names],names
        self.assert_io(eq(x, self.x[names]), fname)

    def test_toload(self):
        fname = TestDataDir + 'toload.tsv'
        self.x.saveSV(fname)
        names=[self.x.dtype.names[i] for i in [0,-1]]
        x = tb.tabarray(SVfile=fname, usecols=names)
        self.assert_io(eq(x, self.x[names]), fname)

    def test_loadSVrecs(self):
        fname = TestDataDir + 'loadSVrecs.tsv'
        self.x.saveSV(fname)
        [recs, metadata] = tb.loadSVrecs(fname)
        names = metadata['names']
        print names
        if 'coloring' in metadata.keys():
            coloring = metadata['coloring']
        else:
            coloring = {}
        self.assert_io(all([recs == self.x.extract().tolist(), 
                       names == list(self.x.dtype.names), 
                       coloring == self.x.coloring]), fname)

# n = ndarray, recarray or tabarray
# F = field name
# C = complex field name
# S = slice
# i = complex index

class TesterGetSet(unittest.TestCase):
# tabarray vs recarray vs array

    def setUp(self):
        X = [(1, 'a', 4, 'ccc', 3.0), (2, 'b', 5, 'd', 4.0), 
             (7, 'e', 2, 'j', 8.0), (2, 'e', 2, 'j', 3.0)]
        names=['f' + str(i) for i in range(len(X[0]))]
        formats='i4,|S1,i4,|S3,f8'.split(',')
        dtype = np.dtype({'names': names, 'formats': formats})
        self.A = np.array(X, dtype=dtype)
        self.R = np.rec.fromrecords(X)
        self.D = tb.tabarray(records=X)
        self.A1 = self.A.copy()
        self.R1 = self.R.copy()
        self.D1 = self.D.copy()

    def test_eq(self):
        self.assert_(eq3(self))

    def test_SF(self):
        # n[S][F] = something -- resets n
        x = self.A[1:]['f2']
        self.A[1:]['f2'] = x + 1
        self.R[1:]['f2'] = x + 1
        self.D[1:]['f2'] = x + 1
        self.assert_(neq3(self))

    def test_FS(self):
        # n[F][S] = something -- resets n
        x = self.A['f2'][1:]
        self.A['f2'][1:] = x + 1
        self.R['f2'][1:] = x + 1
        self.D['f2'][1:] = x + 1
        self.assert_(neq3(self))

    def test_Fi(self):
        # n[F][i] = something -- resets n
        x = self.A['f2'][[1,3]]
        self.A['f2'][[1,3]] = x + 1
        self.R['f2'][[1,3]] = x + 1
        self.D['f2'][[1,3]] = x + 1
        self.assert_(neq3(self))

    def test_iF(self):
        # n[i][F] = something -- does nothing
        x = self.A[[1,3]]['f2']
        self.A[[1,3]]['f2'] = x + 1
        self.R[[1,3]]['f2'] = x + 1
        self.D[[1,3]]['f2'] = x + 1
        self.assert_(eq3(self))

    def test_FS_(self):
        # V = n[F][S], V = something -- does nothing
        x = self.A.copy()['f2'][1:]
        A = self.A['f2'][1:]
        R = self.R['f2'][1:]
        D = self.R['f2'][1:]
        A = x + 1
        R = x + 1
        D = x + 1
        self.assert_(eq3(self))

    def test_Fi_(self):
        # V = n[F][i], V = something -- does nothing
        x = self.A.copy()['f2'][[1,3]]
        A = self.A['f2'][[1,3]]
        R = self.R['f2'][[1,3]]
        D = self.R['f2'][[1,3]]
        A = x + 1
        R = x + 1
        D = x + 1
        self.assert_(eq3(self))

    def test_iF_(self):
        # V = n[i][F], V = something -- does nothing
        x = self.A.copy()[[1,3]]['f2']
        A = self.A[[1,3]]['f2']
        R = self.R[[1,3]]['f2']
        D = self.R[[1,3]]['f2']
        A = x + 1
        R = x + 1
        D = x + 1
        self.assert_(eq3(self))

    def test_Fi_S(self):
        # V = n[F][i], V[S] = something -- does nothing
        x = self.A.copy()['f2'][[1,3]]
        A = self.A['f2'][[1,3]]
        R = self.R['f2'][[1,3]]
        D = self.R['f2'][[1,3]]
        A[:] = x + 1
        R[:] = x + 1
        D[:] = x + 1
        self.assert_(eq3(self))

    def test_FS_S(self):
        # V = n[F][S], V[S] = something -- resets n
        x = self.A.copy()['f2'][1:]
        A = self.A['f2'][1:]
        R = self.R['f2'][1:]
        D = self.D['f2'][1:]
        A[:] = x + 1
        R[:] = x + 1
        D[:] = x + 1
        self.assert_(neq3(self))

    def test_FS_i(self):
        # V = n[F][S], V[i] = something -- resets n
        x = self.A.copy()['f2'][1:]
        A = self.A['f2'][1:]
        R = self.R['f2'][1:]
        D = self.D['f2'][1:]
        i = range(len(A)-1)
        A[i] = x + 1
        R[i] = x + 1
        D[i] = x + 1
        self.assert_(neq3(self))


class TesterGetSet_Big(TesterGetSet):

    def setUp(self):
        X = np.random.rand(10 ** 3, 10 ** 2)
        names=['f' + str(i) for i in range(len(X[0]))]
        formats=['f8']*len(X[0])
        dtype = np.dtype({'names': names, 'formats': formats})
        self.A = np.array([tuple(row) for row in X], dtype=dtype)
        self.R = np.rec.fromrecords(X)
        self.D = tb.tabarray(array=X)
        self.A1 = self.A.copy()
        self.R1 = self.R.copy()
        self.D1 = self.D.copy()


def TestExtract():
    D = tb.tabarray(array = np.random.rand(10 ** 3 + 50, 10 ** 2))
    assert(isinstance(np.sum(D.extract()), float))

def TestReplace():
    V1 = ['North', 'South', 'East', 'West']
    V2 = ['Service', 'Manufacturing', 'Education', 'Healthcare']
    Recs = [(a, b, np.random.rand() * 100, np.random.randint(100000)) 
            for a in V1 for b in V2]
    Recs2 = [(a, b.replace('Education','Taxes'), c, d) for (a, b, c, d) 
                                                                       in Recs]
    X = tb.tabarray(records=Recs, 
                    names=['Region', 'Sector', 'Amount', 'Population'])
    X2 = tb.tabarray(records=Recs2,
                    names=['Region', 'Sector', 'Amount', 'Population'])

    X.replace('Education', 'Taxes')
    assert((X == X2).all())

def TestReplace2():
    V1 = ['North', 'South', 'East', 'West']
    V2 = ['Service', 'Manufacturing', 'Education', 'Healthcare']
    Recs = [(a, b, np.random.rand() * 100, np.random.randint(100000)) 
                                                       for a in V1 for b in V2]
    X = tb.tabarray(records=Recs, 
                    names=['Region', 'Sector', 'Amount', 'Population'])
    X2 = tb.tabarray(records=Recs, 
                     names=['Region', 'Sector', 'Amount', 'Population'])

    X.replace('S', 'M')
    assert((X == X2).all())

def TestReplace3():
    V1 = ['North', 'South', 'East', 'West']
    V2 = ['Service', 'Manufacturing', 'Education', 'Healthcare']
    Recs = [(a, b, np.random.rand() * 100, np.random.randint(100000)) 
                                                       for a in V1 for b in V2]
    Recs2 = [(a.replace('e', 'B'), b.replace('e', 'B'), c, d) for (a, b, c, d) 
                                                                       in Recs]
    X = tb.tabarray(records=Recs,
                    names=['Region', 'Sector', 'Amount', 'Population'])
    X2 = tb.tabarray(records=Recs2,
                     names=['Region', 'Sector', 'Amount', 'Population'])

    X.replace('e', 'B', strict=False)
    assert((X == X2).all())

def TestPivot1():
    X = tb.tabarray(records=[('x', 1, 3, 6), ('y', 1, 5, 3), ('y', 0, 3, 1), 
                             ('x', 0, 3, 5)], names=['a', 'b', 'c', 'd'])
    Y = X.pivot('b', 'a')
    Z = tb.tabarray(records=[(0, 3, 3, 5, 1), (1, 3, 5, 6, 3)], 
                    names=['b', 'x_c', 'y_c', 'x_d', 'y_d'])
    assert (Y == Z).all()

def TestPivot2():
    X = tb.tabarray(records=[('x', 1, 3, 6), ('y', 0, 3, 1), ('x', 0, 3, 5)], 
                    names=['a', 'b', 'c', 'd'])
    Y = X.pivot('b', 'a')
    Z = tb.tabarray(records=[(0, 3, 3, 5, 1), (1, 3, 0, 6, 0)], 
                    names=['b', 'x_c', 'y_c', 'x_d', 'y_d'])
    assert (Y == Z).all()

def TestPivot3():
    V1 = ['NorthAmerica', 'SouthAmerica', 'Europe', 'Asia', 'Australia', 
          'Africa', 'Antarctica']
    V1.sort()
    V2 = ['House', 'Car', 'Boat', 'Savings', 'Food', 'Entertainment', 'Taxes']
    V2.sort()
    Recs = [(a, b, 100 * np.random.rand()) for a in V1 for b in V2]
    X = tb.tabarray(records=Recs, names=['Region', 'Source', 'Amount'])
    Y = X.pivot('Region', 'Source')
    Z = utils.uniqify(X['Source'])
    Z.sort()
    Cols = [[y['Amount'] for y in X if y['Source'] == b] for b in Z]
    W = tb.tabarray(columns=[V1] + Cols, 
                    names=['Region'] + [b + '_Amount' for b in Z])
    assert (W == Y).all()

def TestPivot4():
    V1 = ['NorthAmerica', 'SouthAmerica', 'Europe', 'Asia', 'Australia', 'Africa', 'Antarctica']
    V1.sort()
    V2 = ['House', 'Car', 'Boat', 'Savings', 'Food', 'Entertainment', 'Taxes']
    V2.sort()
    Recs = [(a, b, 100 * np.random.rand()) for a in V1 for b in V2]
    X = tb.tabarray(records=Recs[:-1],
                    names=['Region', 'Source', 'Amount'])
    Y = X.pivot('Region', 'Source', 
                NullVals=dict([(o,-999) for o in X.dtype.names]))
    X2 = tb.tabarray(records=Recs, names=['Region', 'Source', 'Amount'])
    Y2 = X.pivot('Region','Source')
    Y2[V2[-1] + '_Amount'][-1] = -999

    assert (Y == Y2).all()

    

from tabular.utils import uniqify,listunion,perminverse

def TestUniqify():
    Input = [2, 3, 4, 4, 4, 5, 5, 1, 1, 2, 3, 6, 6, 5]
    Output = [2, 3, 4, 5, 1, 6]
    assert (Output == uniqify(Input))

def TestUniqify2():
    Input = [2, 3, 4, 4, 4, 5, 5, 1, 1, 2, 3, 6, 6, 5]
    Output = [2, 3, 4, 5, 1, 6]
    assert (Output == uniqify(Input))

def Testlistunion():
    Input = [[2, 3, 4], [4, 5, 6], [6, 4, 2]]
    Output = [2, 3, 4, 4, 5, 6, 6, 4, 2]
    assert (Output == listunion(Input))

def Testperminverse():
    X = np.random.randint(0, 10000, size=(5000,))
    s = X.argsort()
    assert (s[perminverse(s)] == np.arange(len(X))).all()

from tabular.fast import *

def TestArrayUniqify():
    A = np.array([2, 3, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 8, 9, 9, 8,
                  8, 8, 2, 2, 2, 2, 3, 3, 1, 1, 1, -1])
    [D, s] = arrayuniqify(A)
    C = A[s]
    B = np.array([i for i in range(len(C)) if C[i] not in C[:i]])

    ind = arrayuniqify(A, retainorder=True)
    E = [i for i in range(len(A)) if A[i] not in A[:i]]

    return (D.nonzero()[0] == B).all()  and (ind == E)

def TestRecarrayUniqify():
    A = np.rec.fromrecords([(3, 4, 'b'), (2, 3, 'a'), 
                            (2, 3, 'a'), (3, 4, 'b')], names=['A','B','C'])
    [D, s] = recarrayuniqify(A)
    C = A[s]
    B = np.array([i for i in range(len(C)) if C[i] not in C[:i]])

    ind = arrayuniqify(A, retainorder=True)
    E = [i for i in range(len(A)) if A[i] not in A[:i]]

    return (D.nonzero()[0] == B).all()  and (ind == E)

def TestEqualsPairs():
    N = 100
    Y = np.random.randint(0, 10, size=(N,))
    Y.sort()
    X = np.random.randint(0, 10, size=(N,))

    A = np.array([min((Y == k).nonzero()[0]) for k in X])
    B = np.array([1 + max((Y == k).nonzero()[0]) for k in X])
    [C, D] = equalspairs(X,Y)

    assert (A == C).all() and (B == D).all()

def TestRecarrayEqualsPairs():
    N = 100
    C1 = np.random.randint(0, 3, size=(N,))
    ind = np.random.randint(0, 3, size=(N,))
    v = np.array(['a', 'b', 'c'])
    C2 = v[ind]
    Y = np.rec.fromarrays([C1, C2], names=['A', 'B'])

    C1 = np.random.randint(0, 3, size=(N,))
    ind = np.random.randint(0, 3, size=(N,))
    v = np.array(['a', 'b', 'c'])
    C2 = v[ind]
    X = np.rec.fromarrays([C1, C2], names=['A', 'B'])

    [C, D, s] = recarrayequalspairs(X, Y)
    Y = Y[s]
    A = np.array([min((Y == k).nonzero()[0]) for k in X])
    B = np.array([1 + max((Y == k).nonzero()[0]) for k in X])

    assert (A == C).all() and (B == D).all()

def TestIsIn():
    Y = np.random.randint(0, 10000, size=(100,))
    X = np.arange(10000)
    Z = isin(X, Y)
    D = np.array(uniqify(Y))
    D.sort()
    T1 = (X[Z] == D).all()

    X = np.array(range(10000) + range(10000))
    Z = isin(X, Y)
    T2 = (X[Z] == np.append(D, D.copy())).all()

    assert T1 & T2


def nullvalue(test):
    return False if isinstance(test, bool) \
           else 0 if isinstance(test, int) \
           else 0.0 if isinstance(test, float) \
           else ''

def eq(x,y):
    """
    this special definition of equality is used for checking equality of 
    arrays that might have NaNs. 
    """
    if x.dtype != y.dtype:
        return False
    else:
        names = x.dtype.names        
        for a in names:
            try:
                b = ((np.isnan(x[a]) == np.isnan(y[a])) | (x[a] == y[a])).all()
            except NotImplementedError:
                b = (x[a] == y[a]).all()
            if b is False:
            	return b        
        return ColorEq(x, y)


def ColorEq(x, y):
    return x.coloring.keys() == y.coloring.keys()  and \
         all([x[k].dtype.names == y[k].dtype.names for k in x.coloring.keys()])
         

def eq3(X):
    return (all(X.A == X.A1) and all(X.R == X.R1) and all(X.D == X.D1))

def neq3(X):
    return (any((X.A != X.A1)) and any((X.R != X.R1)) and any((X.D != X.D1)))
