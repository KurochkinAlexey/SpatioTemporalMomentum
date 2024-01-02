from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import abc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
class TrainValTestSplitter(abc.ABC):
    
    @abc.abstractmethod
    def split(self, start, val_delta, test_delta, seed):
        pass
    

class MultivariateTrainValTestSplitter(TrainValTestSplitter):
    def __init__(self, data, cols, datetime_cols, target_col, orig_returns_col, vol_col,
                 timesteps=63, scaling=None, batch_size=64):
        self._data = deepcopy(data)
        self._cols = cols
        self._datetime_cols = datetime_cols
        self._target_col = target_col
        self._orig_returns_col = orig_returns_col
        self._vol_col = vol_col
        self._scaling = scaling
        self._scalers = {}
        self._timesteps = timesteps
        self._batch_size= batch_size
        
        assert len(datetime_cols) == 0
        assert self._target_col not in self._cols
        assert self._orig_returns_col not in self._cols
       # assert self._vol_col not in self._cols
        
        
    def split(self, start, val_delta, test_delta, seed):        
        
        offset_delta = pd.Timedelta('1day')
        
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        y_train_orig, y_val_orig, y_test_orig = [], [], []
        vol_train, vol_val, vol_test = [], [], []
        
        test_datetimes = []
        datetime_features_created = False
        
        for key in self._data.keys():
            self._data[key]['idx'] = np.arange(len(self._data[key]))
            train_val_test = self._data[key].loc[:start+val_delta+test_delta]
            
            
            # optionally scale features
            if self._scaling is not None:
                if self._scaling == 'minmax':
                    scaler = MinMaxScaler().fit(train_val_test.loc[:start, self._cols])
                elif self._scaling == 'standard':
                    scaler = StandardScaler().fit(train_val_test.loc[:start, self._cols])
                
                else:
                    raise NotImplementedError
                
                self._scalers[(start, key)] = scaler

                train_val_test.loc[:, self._cols] = scaler.transform(train_val_test.loc[:, self._cols])


            X = np.zeros((len(train_val_test), self._timesteps, len(self._cols)))
            y = np.zeros((len(train_val_test), self._timesteps, 1))

            #collect non scaled target returns data for further model evaluation
            y_orig = np.zeros((len(train_val_test), self._timesteps, 1))
            #collect volatility data for turnover regularization and/or evaluation
            vol = np.zeros((len(train_val_test), self._timesteps, 1))
            
            for i, col in enumerate(self._cols):
                for j in range(self._timesteps):
                    X[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1)
                    

            for j in range(self._timesteps):
                y[:, j, 0] = train_val_test[self._target_col].shift(self._timesteps - j - 1)
                y_orig[:, j, 0] = train_val_test[self._orig_returns_col].shift(self._timesteps - j - 1)
                vol[:, j, 0] = train_val_test[self._vol_col].shift(self._timesteps - j - 1)
                
            
            train_idx = train_val_test.loc[:start, 'idx']
            val_idx = train_val_test.loc[start+offset_delta:start+val_delta, 'idx']
            test_idx = train_val_test.loc[start+val_delta+offset_delta:start+val_delta+test_delta, 'idx']
            
            val_dt = train_val_test.loc[train_val_test['idx'].isin(val_idx)].index
            test_dt = train_val_test.loc[train_val_test['idx'].isin(test_idx)].index
            test_datetimes.append(test_dt)
            
            X_train_, y_train_, y_train_orig_, vol_train_ = \
                                X[train_idx], y[train_idx], y_orig[train_idx], vol[train_idx]
            X_val_, y_val_, y_val_orig_, vol_val_ = \
                                X[val_idx], y[val_idx], y_orig[val_idx], vol[val_idx]
            
            X_test_, y_test_, y_test_orig_, vol_test_ = \
                                X[test_idx], y[test_idx], y_orig[test_idx], vol[test_idx]
            
            X_train_ = X_train_[self._timesteps:]
            y_train_ = y_train_[self._timesteps:]
            y_train_orig_ = y_train_orig_[self._timesteps:]
            vol_train_ = vol_train_[self._timesteps:]
            
            X_train.append(X_train_)
            X_val.append(X_val_)
            X_test.append(X_test_)
            
            y_train.append(y_train_)
            y_val.append(y_val_)
            y_test.append(y_test_)
            
            y_train_orig.append(y_train_orig_)
            y_val_orig.append(y_val_orig_)
            y_test_orig.append(y_test_orig_)
            
            vol_train.append(vol_train_)
            vol_val.append(vol_val_)
            vol_test.append(vol_test_)
            
            self._data[key] = self._data[key].drop(['idx'], axis=1)
            
        arrays = [X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig, 
                  vol_train, vol_val, vol_test]
            
        def _to_tensor(x):
            x = np.concatenate(x, axis=2)
            x = torch.Tensor(x)
            return x
        
        for i in range(len(arrays)):
            arrays[i] = _to_tensor(arrays[i])
        
        X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig, \
                  vol_train, vol_val, vol_test = arrays
        
        #check alignment by time axis

        for i in range(1, len(test_datetimes)):
            assert np.all((test_datetimes[i] - test_datetimes[0]) == pd.Timedelta(0))
        assert len(X_test) == len(test_datetimes[0]) 

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(TensorDataset(X_train, y_train, y_train_orig, vol_train),
                                  shuffle=True, batch_size=self._batch_size,
                                 worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(TensorDataset(X_val, y_val, y_val_orig, vol_val),
                                shuffle=False, batch_size=self._batch_size)
        test_loader = DataLoader(TensorDataset(X_test, y_test, y_test_orig, vol_test),
                                 shuffle=False, batch_size=self._batch_size)
        
        return train_loader, val_loader, test_loader, test_datetimes[0]