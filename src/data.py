from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import abc


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
class TrainValTestSplitter(abc.ABC):
    
    @abc.abstractmethod
    def split(self, start, val_delta, test_delta, seed):
        pass
    
# data format used in https://arxiv.org/pdf/2302.10175.pdf
class MultivariateTrainValTestSplitter(TrainValTestSplitter):
    def __init__(self, data, cols, cat_cols, target_col, orig_returns_col, vol_col,
                 timesteps=63, scaling=None, batch_size=64, encoder_length=None):
        self._data = deepcopy(data)
        self._cols = cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._orig_returns_col = orig_returns_col
        self._vol_col = vol_col
        self._scaling = scaling
        self._scalers = {}
        self._timesteps = timesteps
        self._batch_size= batch_size
        self._encoder_length = encoder_length
        
        # to avoid data leak
        assert self._target_col not in self._cols
        assert self._orig_returns_col not in self._cols
        if encoder_length is not None:
            assert encoder_length < timesteps
        
        
    def split(self, start, val_delta, test_delta, seed):        
        
        # used to avoid overlap of train/val/test in pandas .loc
        offset_delta = pd.Timedelta('1day')
        
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        y_train_orig, y_val_orig, y_test_orig = [], [], []
        vol_train, vol_val, vol_test = [], [], []
        
        test_datetimes = []
        
        for key in self._data.keys():
            # used to split data into train/val/test
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

            cols = self._cols
            
            target_timesteps = self._timesteps
            if self._encoder_length is not None:
                # in case of tft model, create target lookback equal to length of decoder output
                target_timesteps -= self._encoder_length
            
            X = np.zeros((len(train_val_test), self._timesteps, len(cols)))
            y = np.zeros((len(train_val_test), target_timesteps, 1))

            #collect non scaled target returns data for further model evaluation
            y_orig = np.zeros((len(train_val_test), target_timesteps, 1))
            #collect volatility data for turnover regularization and/or evaluation
            vol = np.zeros((len(train_val_test), target_timesteps, 1))
            
            for i, col in enumerate(cols):
                for j in range(self._timesteps):
                    X[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1)
                    

            for j in range(target_timesteps):
                y[:, j, 0] = train_val_test[self._target_col].shift(target_timesteps - j - 1)
                y_orig[:, j, 0] = train_val_test[self._orig_returns_col].shift(target_timesteps - j - 1)
                vol[:, j, 0] = train_val_test[self._vol_col].shift(target_timesteps - j - 1)
                
            # train/val/test split
            # note that there is overlap equal to self._timesteps between folds
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
            
            # remove nan samples after applying pd.Series.shift
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
            
            # remove auxiliary data
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
        
        # optionally create categorical data tensors
        if self._cat_cols:
            X_cat = np.zeros((len(train_val_test), self._timesteps, len(self._cat_cols)))
            
            for i, col in enumerate(self._cat_cols):
                for j in range(self._timesteps):
                    X_cat[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1)
            
            X_train_cat, X_val_cat, X_test_cat =\
                torch.Tensor(X_cat[train_idx]), torch.Tensor(X_cat[val_idx]), torch.Tensor(X_cat[test_idx])
            
            X_train_cat = X_train_cat[self._timesteps:]
            
        #check alignment by time axis
        for i in range(1, len(test_datetimes)):
            assert np.all((test_datetimes[i] - test_datetimes[0]) == pd.Timedelta(0))
        assert len(X_test) == len(test_datetimes[0]) 
        
        # fix batches sampling order for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        
        if self._encoder_length is None:
            # dataloaders for non tft models
            if self._cat_cols:
                # note that categorical variables should be scaled in case of big absolute maximum values
                X_train = torch.cat([X_train, X_train_cat], dim=2)
                X_val = torch.cat([X_val, X_val_cat], dim=2)
                X_test = torch.cat([X_test, X_test_cat], dim=2)
            
            cat_info = {}
            
            train_loader = DataLoader(TensorDataset(X_train, y_train, y_train_orig, vol_train),
                                  shuffle=True, batch_size=self._batch_size,
                                 worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, y_val, y_val_orig, vol_val),
                                    shuffle=False, batch_size=self._batch_size)
            test_loader = DataLoader(TensorDataset(X_test, y_test, y_test_orig, vol_test),
                                     shuffle=False, batch_size=self._batch_size)
        else:
            # in case of tft model, input tensor should be split into encoder and decoder parts
            def _split_encoder_decoder(x, t):
                x_enc = x[:, :t, :]
                x_dec = x[:, t:, :]

                x_enc_length = torch.ones(len(x)).long()
                x_enc_length.fill_(x_enc.shape[1])
                x_dec_length = torch.ones(len(x)).long()
                x_dec_length.fill_(x_dec.shape[1])
            
                return x_enc, x_dec, x_enc_length, x_dec_length
            
            
            X_train_enc_real, X_train_dec_real, X_train_enc_len, X_train_dec_len =\
                _split_encoder_decoder(X_train, self._encoder_length)
            
            X_val_enc_real, X_val_dec_real, X_val_enc_len, X_val_dec_len =\
                _split_encoder_decoder(X_val, self._encoder_length)
            
            X_test_enc_real, X_test_dec_real, X_test_enc_len, X_test_dec_len =\
                _split_encoder_decoder(X_test, self._encoder_length)
            
            
            if not self._cat_cols:
                # if no categorical features are used, create dummy tensors for tft categorical inputs
                X_train_cat = torch.zeros_like(X_train).long()
                X_val_cat = torch.zeros_like(X_val).long()
                X_test_cat = torch.zeros_like(X_test).long()
            else:
                X_train_cat = X_train_cat.long()
                X_val_cat = X_val_cat.long()
                X_test_cat = X_test_cat.long()
            
            X_train_enc_cat, X_train_dec_cat, _, _ =\
                _split_encoder_decoder(X_train_cat, self._encoder_length)
            
            X_val_enc_cat, X_val_dec_cat, _, _ =\
                _split_encoder_decoder(X_val_cat, self._encoder_length)
            
            X_test_enc_cat, X_test_dec_cat, _, _ =\
                _split_encoder_decoder(X_test_cat, self._encoder_length)
            
            
            
            train_loader = DataLoader(TensorDataset(X_train_enc_real, X_train_enc_cat,
                                                    X_train_dec_real, X_train_dec_cat,
                                                    X_train_enc_len, X_train_dec_len,
                                                    y_train, y_train_orig, vol_train),
                                                    shuffle=True, batch_size=self._batch_size,
                                                    worker_init_fn=seed_worker, generator=g)
            
            val_loader = DataLoader(TensorDataset(X_val_enc_real, X_val_enc_cat,
                                                  X_val_dec_real, X_val_dec_cat,
                                                  X_val_enc_len, X_val_dec_len,
                                                  y_val, y_val_orig, vol_val),
                                                  shuffle=False, batch_size=self._batch_size)
            
            test_loader = DataLoader(TensorDataset(X_test_enc_real, X_test_enc_cat,
                                                   X_test_dec_real, X_test_dec_cat,
                                                   X_test_enc_len, X_test_dec_len,
                                                   y_test, y_test_orig, vol_test),
                                                   shuffle=False, batch_size=self._batch_size)
            
            # collect number of unique entries in each categorical feature
            if self._cat_cols:
                cat_info = {i: len(torch.unique(X_train_enc_cat[..., i])) \
                            for i in range(len(self._cat_cols))}
            else:
                cat_info = {}

                
        return train_loader, val_loader, test_loader, test_datetimes[0], cat_info
    

# data format used in https://arxiv.org/pdf/2112.08534.pdf
class UnivariateTrainValTestSplitter(TrainValTestSplitter):
    def __init__(self, data, cols, cat_cols, target_col, orig_returns_col, vol_col,
                 timesteps=63, scaling=None, batch_size=256, encoder_length=None, use_asset_info_as_feature=False):
        self._data = deepcopy(data)
        self._cols = cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._orig_returns_col = orig_returns_col
        self._vol_col = vol_col
        self._scaling = scaling
        self._scalers = {}
        self._timesteps = timesteps
        self._batch_size= batch_size
        self._encoder_length = encoder_length
        self._use_asset_info_as_feature = use_asset_info_as_feature
        
        # to avoid data leak
        assert self._target_col not in self._cols
        assert self._orig_returns_col not in self._cols
        if encoder_length is not None:
            assert encoder_length < timesteps
        
        
    def split(self, start, val_delta, test_delta, seed):        
        
        # used to avoid overlap of train/val/test in pandas .loc
        offset_delta = pd.Timedelta('1day')
        
        test_datetimes = []
        
        for key in self._data.keys():
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
        
        # train data is concatenated into single dataset (asset_1, asset_2, ..., asset_n)
        concat_train_df_real, concat_train_df_cat = [], []
        # for evaluation purposes concatenation is not needed
        val_data_real, val_data_cat, test_data_real, test_data_cat = {}, {}, {}, {}
        test_datetimes = []

        for key in self._data.keys():
        
            df = self._data[key]
           
            df[self._cols + [self._target_col, self._vol_col]] =\
                df[self._cols + [self._target_col, self._vol_col]].astype(np.float32)
            
            train_df_ = df.loc[:start]
            val_df_ = df.loc[start+offset_delta:start+val_delta]
            test_df_ = df.loc[start+val_delta+offset_delta:start+val_delta+test_delta]
            
            # to align data in TrainDataset and EvalDataset (timedelta should be used insted of index slice)
            train_df = train_df_.iloc[:-self._timesteps]
            val_df = pd.concat([train_df_.iloc[-self._timesteps:], val_df_.iloc[:-self._timesteps]])
            test_df = pd.concat([val_df_.iloc[-self._timesteps:], test_df_])
            
            test_datetime = test_df.index[self._timesteps:]
            
            train_df_real = train_df[self._cols+[self._target_col, self._vol_col, 'asset']]
            val_df_real = val_df[self._cols+[self._target_col, self._vol_col, 'asset']]
            test_df_real = test_df[self._cols+[self._target_col, self._vol_col, 'asset']]
            
            concat_train_df_real.append(train_df_real)
            val_data_real[key] = val_df_real
            test_data_real[key] = test_df_real
            
            # optionally use categorical features
            if self._cat_cols:
                train_df_cat, val_df_cat, test_df_cat =\
                    train_df[self._cat_cols], val_df[self._cat_cols], test_df[self._cat_cols]
                concat_train_df_cat.append(train_df_cat)
                val_data_cat[key] = val_df_cat
                test_data_cat[key] = test_df_cat
                
            test_datetimes.append(test_datetime)
        
        #check alignment by time axis

        for i in range(1, len(test_datetimes)):
            assert np.all((test_datetimes[i] - test_datetimes[0]) == pd.Timedelta(0))
            
        concat_train_df_real = pd.concat(concat_train_df_real)
            
            
        # use asset name info to avoid overlapping of different assets features during training
        # (case when asset_n-1 and asset_n data is in the single batch during training)
        le = LabelEncoder()
        concat_train_df_real['label'] = le.fit_transform(concat_train_df_real['asset'].values)
        for key in val_data_real.keys():
            val_data_real[key]['label'] = le.transform(val_data_real[key]['asset'].values)
            test_data_real[key]['label'] = le.transform(test_data_real[key]['asset'].values)
        concat_train_df_real['label_diff'] = concat_train_df_real['label'].diff().bfill()
        
        # collect number of unique entries in categorial features
        cat_info = {}
        if concat_train_df_cat:
            concat_train_df_cat = pd.concat(concat_train_df_cat)
            cat_info = {i: concat_train_df_cat[self._cat_cols[i]].nunique() for i in range(len(self._cat_cols))}
        if self._use_asset_info_as_feature:
            # case if asset name is used as categorical feature
            n_assets = concat_train_df_real['label'].nunique()
            if cat_info:
                max_key = max(cat_info.keys())
                cat_info[max_key+1] = n_assets
            else:
                cat_info[0] = n_assets
        
        # train and eval datasets are treated differently
        train_dataset = TrainDataset(concat_train_df_real, concat_train_df_cat,
                                     self._timesteps, self._encoder_length,
                                     self._cols, self._cat_cols,
                                     self._target_col, self._vol_col,
                                     use_asset_info_as_feature=self._use_asset_info_as_feature)
        
        val_dataset = EvalDataset(val_data_real, val_data_cat,
                                  self._timesteps, self._encoder_length,
                                  self._cols, self._cat_cols,
                                  self._target_col, self._vol_col,
                                  use_asset_info_as_feature=self._use_asset_info_as_feature)
        
        test_dataset = EvalDataset(test_data_real, test_data_cat,
                                  self._timesteps, self._encoder_length,
                                  self._cols, self._cat_cols,
                                  self._target_col, self._vol_col,
                                  use_asset_info_as_feature=self._use_asset_info_as_feature)
        
        # fix batches sampling order for reproducibility
        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(train_dataset,
                                  shuffle=True, batch_size=self._batch_size,
                                 worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset,
                                shuffle=False, batch_size=self._batch_size)
        test_loader = DataLoader(test_dataset,
                                 shuffle=False, batch_size=self._batch_size)
        
        return train_loader, val_loader, test_loader, test_datetimes[0], cat_info
    
    
class TrainDataset(Dataset):
    
    def __init__(self, df_real, df_cat,
                 history_size, encoder_length,
                 real_cols, cat_cols,
                 target_col, vol_col,
                 use_asset_info_as_feature):
        
        self._df_real = df_real
        self._df_cat = df_cat
        self._history_size = history_size
        self._encoder_length = encoder_length
        self._real_cols = real_cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._vol_col = vol_col
        self._use_asset_label = use_asset_info_as_feature
    
    
    def __len__(self):
        # exclude lookback window from dataset size
        return max(len(self._df_real) - self._history_size, 0)
    
    
    def __getitem__(self, idx):
        
        # slice lookback window
        X_real = self._df_real.iloc[idx:idx+self._history_size][self._real_cols].values
        
        # create batch of categorical data
        X_cat = None
        if self._use_asset_label:
            X_cat = self._df_real.iloc[idx:idx+self._history_size]['label'].values[..., np.newaxis].astype(np.int64)
            
        if self._cat_cols:
            X_cat_ = self._df_cat.iloc[idx:idx+self._history_size][self._cat_cols].values.astype(np.int64)
            if self._use_asset_label:
                X_cat = np.concatenate([X_cat_, X_cat], axis=-1)
            else:
                X_cat = X_cat_
        
        # concatenate categorical data and real data in case of non tft model
        if X_cat is not None and self._encoder_length is None:
            X = np.concatenate([X_real, X_cat.astype(np.float32)], axis=-1)
            
        elif X_cat is None and self._encoder_length is None:
            X = X_real
        
        y = self._df_real.iloc[idx:idx+self._history_size][self._target_col].values[..., np.newaxis]

        vol = self._df_real.iloc[idx:idx+self._history_size][self._vol_col].values[..., np.newaxis]
        
        # mask data where assets are overlapped
        asset_change = self._df_real.iloc[idx:idx+self._history_size]['label_diff'].values

        mask = np.ones_like(y)

        nonzero_entry = np.argwhere(asset_change != 0).ravel()
        if len(nonzero_entry) != 0:
            mask[:, :nonzero_entry[0]] = 0
        
        if self._encoder_length is None:

            return X, y, mask, vol
        
        else:
            # in case of tft model, data should be splitted for encoder and decoder parts
            X_enc_real, X_dec_real = X_real[:self._encoder_length], X_real[self._encoder_length:]
            if X_cat is not None:
                X_enc_cat, X_dec_cat = X_cat[:self._encoder_length], X_cat[self._encoder_length:]
            else:
                # create dummy tensors for tft categorical inputs if categorical features are not used
                X_enc_cat, X_dec_cat = np.zeros(len(X_enc_real)), np.zeros(len(X_dec_real))

            enc_length = len(X_enc_real)
            dec_length = len(X_dec_real)
            
            # target data size is equal to decoder length
            y = y[self._encoder_length:]
            vol = vol[self._encoder_length:]
            mask = mask[self._encoder_length]
            
            return X_enc_real, X_enc_cat, X_dec_real, X_dec_cat, enc_length, dec_length, y, mask, vol
        
        
class EvalDataset(Dataset):
    
    def __init__(self, data_real, data_cat,
                 history_size, encoder_length,
                 real_cols, cat_cols,
                 target_col, vol_col,
                 use_asset_info_as_feature):
        
        self._keys = list(data_real.keys())
        self._data_real = data_real
        self._data_cat = data_cat
        self._history_size = history_size
        self._encoder_length = encoder_length
        self._real_cols = real_cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._vol_col = vol_col
        self._use_asset_label = use_asset_info_as_feature
        
        
    def __len__(self,):
        # exclude lookback window from dataset size
        return max(len(self._data_real[self._keys[0]]) - self._history_size, 0)
    
    
    def __getitem__(self, idx):
        
        # store assets data in dictionaries
        X, X_real, y, vol = {}, {}, {}, {}
        X_enc_real, X_enc_cat, X_dec_real, X_dec_cat, enc_length, dec_length = {}, {}, {}, {}, {}, {}
        for key in self._keys:
            # slice lookback window
            X_real[key] = self._data_real[key].iloc[idx:idx+self._history_size][self._real_cols].values
            y[key] = self._data_real[key].iloc[idx:idx+self._history_size][self._target_col].values[..., np.newaxis]

            vol[key] = self._data_real[key].iloc[idx:idx+self._history_size][self._vol_col].values[..., np.newaxis]
            
            # create categorical features tensor
            X_cat = None
            if self._use_asset_label:
                X_cat = self._data_real[key].iloc[idx:idx+self._history_size]['label'].values[..., np.newaxis].astype(np.int64)

            if self._cat_cols:
                X_cat_ = self._data_cat[key].iloc[idx:idx+self._history_size][self._cat_cols].values.astype(np.int64)
                if self._use_asset_label:
                    X_cat = np.concatenate([X_cat_, X_cat], axis=-1)
                else:
                    X_cat = X_cat_
            
            # concatenate categorical data and real data in case of non tft model
            if X_cat is not None and self._encoder_length is None:
                X[key] = np.concatenate([X_real[key], X_cat.astype(np.float32)], axis=-1)
            elif X_cat is None and self._encoder_length is None:
                X[key] = X_real[key]
                
            if self._encoder_length is not None:
                # in case of tft model, data should be splitted for encoder and decoder parts
                X_enc_real[key] = X_real[key][:self._encoder_length]
                X_dec_real[key] = X_real[key][self._encoder_length:]
                
                if X_cat is not None:
                    X_enc_cat[key] = X_cat[:self._encoder_length]
                    X_dec_cat[key] = X_cat[self._encoder_length:]
                else:
                    # create dummy tensors for tft categorical inputs if categorical features are not used
                    X_enc_cat[key] = np.zeros(len(X_enc_real[key]))
                    X_dec_cat[key] = np.zeros(len(X_dec_real[key]))
                
                enc_length[key] = len(X_enc_real[key])
                dec_length[key] = len(X_dec_real[key])
                
                # target data size is equal to decoder length
                y[key] = y[key][self._encoder_length:]
                vol[key] = vol[key][self._encoder_length:]
                
        if self._encoder_length is None:
            return X, y, vol
        else:
            return X_enc_real, X_enc_cat, X_dec_real, X_dec_cat, enc_length, dec_length, y, vol        
            