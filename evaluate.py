from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

class Evaluator:
    def __init__(self, name):
        self.name = name

        # # meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
        # if not self.name in meta_info:
        #     print(self.name)
        #     error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
        #     error_mssg += 'Available datasets are as follows:\n'
        #     error_mssg += '\n'.join(meta_info.keys())
        #     raise ValueError(error_mssg)

        # self.num_tasks = int(meta_info[self.name]['num tasks'])
        # self.eval_metric = meta_info[self.name]['eval metric']


    def _parse_and_check_input(self, input_dict):
        # if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
        #     if not 'y_true' in input_dict:
        #         RuntimeError('Missing key of y_true')
        #     if not 'y_pred' in input_dict:
        #         RuntimeError('Missing key of y_pred')

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        '''
            y_true: numpy ndarray or torch tensor of shape (num_node, num_tasks)
            y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
        '''

        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        ## check type
        # if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
        #     raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        # if not y_true.shape == y_pred.shape:
        #     raise RuntimeError('Shape of y_true and y_pred must be the same')

        # if not y_true.ndim == 2:
        #     raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        # if not y_true.shape[1] == self.num_tasks:
        #     raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

        return y_true, y_pred

        # else:
        #     raise ValueError('Undefined eval metric %s ' % (self.eval_metric))


    def eval(self, input_dict, t):
        # if self.eval_metric == 'rocauc':
        y_true, y_pred = self._parse_and_check_input(input_dict)
        if t == 'roc':
            return  self._eval_rocauc(y_true, y_pred)
        elif t == 'acc':
            return self._eval_acc(y_true, f(y_pred))
        elif t == 'f1':     
            return self._eval_f1(y_true,f(y_pred))

    # def _evaln_(self, y_true, y_pred):
    #     return {'rocauc': _eval_rocauc(y_true, y_pred), 'acc': _eval_acc(y_true, y_pred)}

    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC and AP score averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        # if len(rocauc_list) == 0:
        #     raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return  sum(rocauc_list)/len(rocauc_list)

    def _eval_f1(self, y_true, y_pred):
        f1_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                f1_list.append(f1_score(y_true[is_labeled,i], y_pred[is_labeled,i]))
        
        return sum(f1_list)/len(f1_list)
            

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return sum(acc_list)/len(acc_list)


def f(x):
    x[x>0] = 1
    x[x<0] = 0
    return x