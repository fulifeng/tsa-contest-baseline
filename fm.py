import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin

class FactorizationMachine(BaseEstimator, ClassifierMixin):
    def __init__(self, tool_path = './', cache_size=None, dim='1,1,8',
                 init_stdev=0.1, iter=100, learn_rate=0.1, method='mcmc',
                 out=None, regular=None, relation=None, rlog=None, task='c',
                 test='', train='', validation=None):
        self.tool_path = tool_path
        self.cache_size = cache_size
        self.dim = dim
        self.init_stdev = init_stdev
        self.iter = iter
        self.learn_rate = learn_rate
        self.method = method
        self.out = out
        self.regular = regular
        self.relation = relation
        self.rlog = rlog
        self.task = task
        self.test = test
        self.validation = validation
        self.train = train

    def fit(self, X, y):
        # here X, y is not used, we just use the data file
        command = [os.path.join(self.tool_path, 'bin/libFM')]
        command.extend(['-dim', self.dim])
        command.extend(['-init_stdev', str(self.init_stdev)])
        command.extend(['-iter', str(self.iter)])
        command.extend(['-learn_rate', str(self.learn_rate)])
        command.extend(['-method', self.method])
        if self.out is None:
            command.extend(['-out', self.test + '-fm_out'])
        else:
            command.extend(['-out', self.out])
        if not self.regular is None:
            command.extend(['-regular', self.regular])
        if not self.relation is None:
            command.extend(['-relation', self.relation])
        if not self.rlog is None:
            command.extend(['-rlog', self.rlog])
        command.extend(['-task', self.task])
        command.extend(['-test', self.test])
        command.extend(['-train', self.train])
        if not self.validation is None:
            command.extend(['-validation', self.validation])
        command_str = ' '.join(command)
        print command_str
        os.system(command_str)
        self.coef_ = list()
        if y is None:
            self.labels = [0, 1]
        else:
            self.labels = np.unique(y)
        return self

    def predict(self, X):
        if not os.path.isfile(self.out):
            return None
        result = np.genfromtxt(self.out, dtype=float, delimiter=',')
        prediction = np.zeros([len(result)], dtype=int)
        for ind, res in enumerate(result):
            if res > 0.5:
                prediction[ind] = self.labels[1]
            else:
                prediction[ind] = self.labels[0]
        return prediction

    def predict_proba(self, X):
        if not os.path.isfile(self.out):
            return None
        result = np.genfromtxt(self.out, dtype=float, delimiter=',')
        prediction = np.zeros([len(result), 2], dtype=float)
        for ind, res in enumerate(result):
            prediction[ind, 0] = res
            prediction[ind, 1] = 1.0 - res
        return prediction

if __name__ == '__main__':
    fm = FactorizationMachine('/home/ffl/nus/MM/cur_trans/tool/libfm',
                              dim='1,1,8', init_stdev=0.005, iter=200,
                              method='mcmc', out='test_fm.out', task='c',
                              test='a1a.t', train='a1a')
    fm.fit(None, None)
    pre_prob = fm.predict_proba(None)
    print pre_prob.shape
    print pre_prob
    pre = fm.predict(None)
    print pre.shape
    print pre