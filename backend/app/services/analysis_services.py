import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, Any, Optional, List
import logging

# --- DEBUG設定: ログを強制的に標準出力に出す ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconometricEngine:
    """
    Econometric Analysis Engine for Macro-Micro Linkage.
    """

    def __init__(self, target_series: pd.Series, exog_series: Optional[pd.DataFrame] = None):
        logger.info("=== [Init] EconometricEngine Initialized ===")
        # 初期化時にバリデーションとソートを実行
        self.y = self._validate_and_set_freq(target_series, name="Target(y)")
        self.exog = self._validate_and_set_freq(exog_series, name="Exog(X)") if exog_series is not None else None
        self.model = None

    def _validate_and_set_freq(self, data: pd.Series | pd.DataFrame, name: str) -> pd.Series | pd.DataFrame:
        if data is None: return None
        data = data.copy()
        
        # --- DEBUG 1: 入力直後の状態を確認 ---
        logger.info(f"--- DEBUG [{name}]: Raw Input ---")
        logger.info(f"Index Type: {type(data.index)}")
        # スライスエラー回避のためチェック
        if len(data) >= 3:
            logger.info(f"Top 3 Index: {data.index[:3].tolist()}")
            logger.info(f"Tail 3 Index: {data.index[-3:].tolist()}")
        else:
            logger.info(f"All Index: {data.index.tolist()}")
        
        # 1. IndexをDatetime型に変換
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
                logger.info(f"[{name}] Converted index to DatetimeIndex.")
            except Exception as e:
                logger.error(f"[{name}] Index conversion failed: {e}")
                raise ValueError(f"Index conversion failed: {e}")
            
        # 2. 値が空でないか確認
        if data.empty:
            raise ValueError(f"Dataset {name} is empty.")

        # 3. 強制ソート (ここが効いているか確認する)
        data.sort_index(ascending=True, inplace=True)

        # 4. 頻度設定 & 欠損処理
        data = data.resample('MS').mean()
        data = data.interpolate(method='linear')
        data = data.astype(float)

        # --- DEBUG 2: 処理完了後の状態を確認 ---
        logger.info(f"--- DEBUG [{name}]: After Processing ---")
        if not data.empty:
            logger.info(f"Start Date (Should be Oldest): {data.index[0]}")
            logger.info(f"End Date (Should be Newest):   {data.index[-1]}")
            logger.info(f"Total Rows: {len(data)}")
            
            # もしEnd Dateが1970年代なら、ここで警告を出す
            if data.index[-1].year < 2000:
                logger.warning(f"⚠️ CRITICAL: Data ends in {data.index[-1].year}! Sorting might be broken or input data is wrong.")

        return data

    def fit(self, seasonal: bool = True, m: int = 12) -> Dict[str, Any]:
        """
        Builds the optimal SARIMA/SARIMAX model.
        """
        y_train = self.y
        exog_train = self.exog
        
        if y_train is not None and not y_train.empty:
            logger.info(f"Fitting model with {len(y_train)} records. Range: {y_train.index[0].date()} -> {y_train.index[-1].date()}")

        self.model = pm.auto_arima(
            y=y_train,
            X=exog_train,
            seasonal=seasonal,
            m=m,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            start_P=0, max_P=1,
            start_Q=0, max_Q=1,
            d=None, D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )

        return {
            "order": self.model.order,
            "seasonal_order": self.model.seasonal_order,
            "aic": self.model.aic(),
            "bic": self.model.bic()
        }

    # ▼▼▼ 復活させたメソッド ▼▼▼
    def diagnose(self) -> Dict[str, Any]:
        if self.model is None: raise RuntimeError("Model not fitted.")
        residuals = self.model.resid()
        # Ensure enough data points for Ljung-Box
        lags = [12] if len(residuals) > 12 else [1]
        lb_df = acorr_ljungbox(residuals, lags=lags, return_df=True)
        lb_pvalue = float(lb_df['lb_pvalue'].iloc[0])
        return {
            "lb_pvalue": lb_pvalue,
            "is_white_noise": lb_pvalue > 0.05,
            "residuals_mean": float(np.mean(residuals)),
            "residuals_std": float(np.std(residuals))
        }

    def predict(self, n_periods: int = 12, future_exog: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if self.model is None: raise RuntimeError("Model not fitted.")
        
        preds, conf_int = self.model.predict(n_periods=n_periods, X=future_exog, return_conf_int=True, alpha=0.05)
        
        # 予測の起点をログ出力
        last_date = self.y.index[-1]
        logger.info(f"--- DEBUG [Predict] ---")
        logger.info(f"Base Date (End of History): {last_date}")
        
        future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq='MS')[1:]
        
        logger.info(f"Forecast Start: {future_dates[0]}")
        logger.info(f"Forecast End:   {future_dates[-1]}")

        return {
            "index": future_dates.strftime('%Y-%m-%d').tolist(),
            "mean": preds.tolist(),
            "lower": conf_int[:, 0].tolist(),
            "upper": conf_int[:, 1].tolist()
        }