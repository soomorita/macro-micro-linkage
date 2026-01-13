import plotly.express as px
import pandas as pd
import json

class VizService:
    def create_time_series_chart(self, df: pd.DataFrame, title: str = "時系列推移"):
        """
        Tidy Dataから時系列折れ線グラフのJSONを生成する
        """
        if df.empty:
            return {}

        # Plotly Express を使用
        # x軸に日付、y軸に値、colorでカテゴリ（品目など）を分ける
        
        # 数値以外の列（ラベル列）を探す
        label_cols = [col for col in df.columns if col not in ['date', 'value', 'unit']]
        color_col = label_cols[0] if label_cols else None

        fig = px.line(
            df,
            x="date",
            y="value",
            color=color_col,
            title=title,
            template="plotly_white",
            labels={"date": "時期", "value": "指数/値"}
        )

        # フロントエンドの Plotly.js で読み込める形式に変換
        return json.loads(fig.to_json())

viz_service = VizService()