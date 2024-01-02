import abc
import pandas as pd
import numpy as np


class AbstractFeatures(abc.ABC):
    
    @abc.abstractmethod
    def calc_features(self, close_srs, open_srs=None, high_srs=None, low_srs=None, volume_srs=None):
        pass


class DeepMomentumFeatures(AbstractFeatures):
    
    def __init__(self, vol_lookback=60, vol_target=0.15, vol_threshold=5, halflife_winsorise=252):
        self._vol_lookback = vol_lookback
        self._vol_target = vol_target
        self._vol_threshold = vol_threshold
        self._halflife_winsorise = halflife_winsorise
        
        
    def _calc_returns(self, srs, day_offset=1, log_transform=False):
        if not log_transform:
            returns = srs / srs.shift(day_offset) - 1.0
        else:
            returns = np.log(srs) - np.log(srs.shift(day_offset).bfill())
        return returns
    
    
    def _calc_daily_vol(self, daily_returns):
        s = daily_returns.ewm(span=self._vol_lookback, min_periods=self._vol_lookback).std().bfill()
        return s
    
    
    def _calc_vol_scaled_returns(self, daily_returns):
        daily_vol = self._calc_daily_vol(daily_returns)
        annualized_vol = daily_vol*np.sqrt(252)
        return daily_returns * self._vol_target / annualized_vol.shift(1)
    
        
    def _calc_normalised_returns(self, df_asset, day_offset):
        return (
            self._calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )
    
    
    def calc_features(self, close_srs, open_srs=None, high_srs=None, low_srs=None, volume_srs=None):
        
        df_asset = pd.DataFrame()
        df_asset['close'] = close_srs
        
        df_asset["srs"] = df_asset["close"]

        df_asset["daily_returns"] = self._calc_returns(df_asset["srs"])
        df_asset["daily_vol"] = self._calc_daily_vol(df_asset["daily_returns"])
        # vol scaling and shift to be next day returns
        df_asset["target_returns"] = self._calc_vol_scaled_returns(
            df_asset["daily_returns"]
        ).shift(-1)
        
        df_asset['target_returns_nonscaled'] = df_asset["daily_returns"].shift(-1)
        
        df_asset["norm_daily_return"] = self._calc_normalised_returns(df_asset, 1)
        df_asset["norm_monthly_return"] = self._calc_normalised_returns(df_asset, 21)
        df_asset["norm_quarterly_return"] = self._calc_normalised_returns(df_asset, 63)
        df_asset["norm_biannual_return"] = self._calc_normalised_returns(df_asset, 126)
        df_asset["norm_annual_return"] = self._calc_normalised_returns(df_asset, 252)
    
        return df_asset#.dropna()

    
class MACDFeatures(AbstractFeatures):
    
    def __init__(self, trend_combinations=[(8, 24), (16, 48), (32, 96)]):
        self._trend_combinations = trend_combinations
        
        
    def _calc_signal(self, srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
        """Calculate MACD signal for a signal short/long timescale combination
        Args:
            srs ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale
        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale):
            return np.log(0.5) / (np.log(1 - 1 / timescale) + 1e-9)

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / (srs.rolling(63).std().bfill() + 1e-9)
        return q / (q.rolling(252).std().bfill() + 1e-9)
    
    
    def calc_features(self, close_srs, open_srs=None, high_srs=None, low_srs=None, volume_srs=None):
        srs = close_srs
        feats = pd.DataFrame()
        
        for comb in self._trend_combinations:
            f = self._calc_signal(srs, comb[0], comb[1])
            feats['macd_{}_{}'.format(comb[0], comb[1])] = f
        
        return feats
    
    
class DatetimeFeatures(AbstractFeatures):
    def __init__(self, **kwargs):
        pass
    
    def calc_features(self, close_srs, open_srs=None, high_srs=None, low_srs=None, volume_srs=None):
        
        df_asset = pd.DataFrame()
        df_asset["day_of_week"] = close_srs.index.dayofweek
        df_asset["day_of_month"] = close_srs.index.day
        df_asset["month_of_year"] = close_srs.index.month
        df_asset = df_asset.set_index(close_srs.index)
        
        return df_asset
    
    
class DefaultFeatureCreator:
    def __init__(self, prices_df, symbols, features, params, half_winsore=252, vol_threshold=5, cut_nan_samples=253):
        self._prices_df = prices_df
        if len(features) == 0:
            raise Exception('Number of features cannot be zero!')
        self._features = features
        self._params = params
        self._half_winsore = half_winsore
        self._vol_threshold = vol_threshold
        self._cut_nan_samples = cut_nan_samples
        self._symbols = symbols
        
        
    def _prepare_series(self, srs, eps=1e-6):
        srs = srs.loc[(srs.notnull()) & (srs > eps)].copy()
        ewm = srs.ewm(halflife=self._half_winsore)
        means = ewm.mean()
        stds = ewm.std()
        srs = np.minimum(srs, means + self._vol_threshold * stds)
        srs = np.maximum(srs, means - self._vol_threshold * stds)
        return srs
        
        
    def create_features(self):
        features = {}
        for symbol in self._symbols:
            prices = {'{}_srs'.format(name): self._prices_df['{}_{}'.format(symbol, name)] for name in \
                     ['open', 'high', 'low', 'close', 'volume']}
            for key in prices.keys():
                prices[key] = self._prepare_series(prices[key])
            features_ = []
            for param, fc in zip(self._params, self._features):
                feature_creator = fc(**param)
                f = feature_creator.calc_features(**prices)
                features_.append(f)
            features_ = pd.concat(features_, axis=1)
            features_.set_index(self._prices_df.index)
            features_ = features_.iloc[self._cut_nan_samples:-1]
            features[symbol] = features_

        return features
                    
