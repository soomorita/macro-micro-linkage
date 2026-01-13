import httpx
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from app.core.config import settings
import logging

# ログ設定: Dockerのログで「どんな日付文字列が来ているか」を見えるようにする
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EStatService:
    def __init__(self):
        self.base_url = settings.ESTAT_BASE_URL
        self.api_key = settings.ESTAT_API_KEY

    async def fetch_stats_data(self, stats_data_id: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        e-Stat APIからデータを取得し、API側のエラーを明示的に検知して例外を投げる。
        """
        query_params = {
            "appId": self.api_key,
            "statsDataId": stats_data_id,
            "metaGetFlg": "Y",
            "cntGetFlg": "N",
        }
        if params:
            query_params.update(params)

        logger.info(f"Requesting e-Stat API: {self.base_url}/getStatsData with params {query_params}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{self.base_url}/getStatsData", params=query_params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"e-Stat API Connection Error: {e}")
                raise ValueError(f"通信エラーが発生しました: {e}")

        # APIレスポンス内のステータスチェック
        root = data.get("GET_STATS_DATA", {})
        result = root.get("RESULT", {})
        status = str(result.get("STATUS", "999"))
        
        if status != "0":
            error_msg = result.get("ERROR_MSG", "Unknown Error")
            logger.error(f"e-Stat API Error: {status} - {error_msg}")
            raise ValueError(f"e-Stat APIエラー [Code {status}]: {error_msg}")

        return self._transform_to_tidy_data(data)

    def _transform_to_tidy_data(self, json_data: Dict[str, Any]) -> pd.DataFrame:
        stat_data = json_data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
        if not stat_data:
            raise ValueError("統計データ(STATISTICAL_DATA)が空です。")

        values = stat_data.get("DATA_INF", {}).get("VALUE", [])
        if isinstance(values, dict):
            values = [values]
            
        df = pd.DataFrame(values)
        if df.empty:
            raise ValueError("取得されたデータが0件です。パラメータ(cat/area)が間違っている可能性があります。")

        class_info = stat_data.get("CLASS_INF", {}).get("CLASS_OBJ", [])
        if isinstance(class_info, dict):
            class_info = [class_info]
            
        df = self._apply_metadata(df, class_info)

        if "$" in df.columns:
            df = df.rename(columns={"$": "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        time_cols = [col for col in df.columns if "時間軸" in col]
        if time_cols:
            # 時間軸カラムを特定
            time_col = time_cols[0]
            df = df.rename(columns={time_col: "date"})
            
            # --- 修正: 強固な日付パースロジック ---
            def parse_date_robust(x):
                s = str(x).strip()
                try:
                    if "月" in s: return pd.to_datetime(s.replace("年", "-").replace("月", "-01"))
                    if "年度" in s: return pd.to_datetime(s.replace("年度", "-04-01"))
                    if "年" in s: return pd.to_datetime(s.replace("年", "-01-01"))
                    
                    # YYYYMM (6桁) -> 202301 -> 2023-01-01
                    if s.isdigit() and len(s) == 6:
                        return pd.to_datetime(s, format="%Y%m")
                    # YYYYMMDD (8桁)
                    if s.isdigit() and len(s) == 8:
                        return pd.to_datetime(s, format="%Y%m%d")
                    # YYYY (4桁)
                    if s.isdigit() and len(s) == 4:
                        return pd.to_datetime(s, format="%Y")
                    
                    return pd.to_datetime(s, errors='coerce')
                except:
                    return pd.NaT
            
            # 生データのサンプルをログに出す（デバッグ用）
            logger.info(f"Raw Date Samples (Before Parse): {df['date'].head().tolist()}")
            
            df["date"] = df["date"].apply(parse_date_robust)
            
            # パース後のチェック
            logger.info(f"Parsed Date Samples: {df['date'].head().tolist()}")
            
            # NaTを除去し、日付で昇順ソート (1970...2024の順に並べる)
            df = df.dropna(subset=["date"]).sort_values("date", ascending=True).reset_index(drop=True)

        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError(f"必要な列(date, value)が見つかりません。Columns: {df.columns.tolist()}")

        return df

    def _apply_metadata(self, df: pd.DataFrame, class_info: List[Dict[str, Any]]) -> pd.DataFrame:
        for obj in class_info:
            col_id, col_name = f"@{obj['@id']}", obj['@name']
            if col_id in df.columns:
                class_list = obj.get("CLASS", [])
                if isinstance(class_list, dict):
                    class_list = [class_list]
                code_to_label = {item["@code"]: item["@name"] for item in class_list if "@code" in item}
                df[col_name] = df[col_id].map(code_to_label)
                df = df.drop(columns=[col_id])
        return df

    def to_wide_format(self, df: pd.DataFrame, index_col: str = "date", columns_col: str = "品目分類", values_col: str = "value") -> pd.DataFrame:
        """
        Tidy Data(縦持ち) を 分析用Wide Format(横持ち) に変換する。
        """
        if df.empty or index_col not in df.columns or values_col not in df.columns:
            return pd.DataFrame()
            
        if columns_col not in df.columns:
            wide_df = df.set_index(index_col)[[values_col]].sort_index()
            wide_df.columns = ["value"]
        else:
            wide_df = df.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc='mean')
            wide_df = wide_df.sort_index()

        # 欠損値処理
        wide_df = wide_df.interpolate(method='linear', limit_direction='both')
        wide_df = wide_df.fillna(method='bfill').fillna(method='ffill')

        return wide_df

estat_services = EStatService()