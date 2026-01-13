import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from typing import Optional
from app.services.estat_services import estat_services
from app.services.viz_services import viz_service
from app.services.analysis_services import EconometricEngine

app = FastAPI(title="Macro-Micro Linkage API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Macro-Micro Linkage Analysis API"}

@app.get("/test-estat/{stats_data_id}")
async def test_estat(stats_data_id: str):
    try:
        df = await estat_services.fetch_stats_data(stats_data_id)
        return {
            "status": "success",
            "row_count": len(df),
            "data": df.head(10).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart/{stats_data_id}")
async def get_chart(
    stats_data_id: str,
    cat: str = "0001",
    area: str = "00000",
):
    """統計データの時系列グラフ(Plotly JSON)を返す"""
    try:
        params = {"cdCat01": cat, "cdArea": area}
        df = await estat_services.fetch_stats_data(stats_data_id, params=params)

        if df.empty:
            return {"status": "success", "message": "条件に一致するデータがありませんでした。", "chart": {}}

        chart_json = viz_service.create_time_series_chart(
            df,
            title=f"e-Stat Data: {stats_data_id} (Cat:{cat}, Area:{area})"
        )

        return {"status": "success", "chart": chart_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{stats_data_id}")
async def get_analysis_data(
    stats_data_id: str,
    area: str = "00000",
):
    """分析用にデータをWide Formatに変換して返す"""
    try:
        params = {"cdArea": area}
        df = await estat_services.fetch_stats_data(stats_data_id, params=params)

        if df.empty:
            return {"status": "error", "message": "データが見つかりませんでした。"}

        cat_cols = [c for c in df.columns if "品目" in c]
        if not cat_cols:
            return {"status": "error", "message": "品目カラムが見つかりませんでした。"}

        target_col = cat_cols[0]
        wide_df = estat_services.to_wide_format(df, columns_col=target_col)
        data_dict = wide_df.reset_index().replace({pd.NaT: None}).to_dict(orient="records")

        return {
            "status": "success",
            "meta": {"rows": len(wide_df), "columns": list(wide_df.columns)},
            "data": data_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_analysis_task(target_series: pd.Series, n_periods: int):
    """Offloads the heavy fit/predict logic to a separate thread."""
    engine = EconometricEngine(target_series=target_series)
    fit_result = engine.fit(seasonal=True, m=12)
    diagnosis = engine.diagnose()
    forecast = engine.predict(n_periods=n_periods)
    return fit_result, diagnosis, forecast

@app.get("/analysis/predict/{stats_data_id}")
async def predict_time_series(
    stats_data_id: str,
    cat: str = "0001",
    area: str = "00000",
    n_periods: int = 12
    ):
    """
    時系列予測(SARIMA)を実行するエンドポイント。
    asyncio.to_thread を使用してブロッキングを回避。
    """
    try:
        # 1. データの非同期取得
        params = {"cdCat01": cat, "cdArea": area}
        df = await estat_services.fetch_stats_data(stats_data_id, params=params)

        if df.empty:
            raise HTTPException(status_code=404, detail="データが見つかりませんでした。")

        # 2. データの整形
        cat_cols = [c for c in df.columns if "品目" in c]
        if not cat_cols:
             raise HTTPException(status_code=500, detail="品目カラムの特定に失敗しました。")
        target_col_name = cat_cols[0]

        wide_df = estat_services.to_wide_format(df, columns_col=target_col_name)
        if wide_df.empty:
             raise HTTPException(status_code=404, detail="Wide Format変換に失敗しました。")

        target_series = wide_df.iloc[:, 0]
        if len(target_series) < 24:
             raise HTTPException(status_code=400, detail=f"データ点数不足({len(target_series)}件)。最低24件必要です。")

        # 3. 重い計算を別スレッドへ委譲（ここでサーバー停止を防ぐ）
        fit_result, diagnosis, forecast = await asyncio.to_thread(
        run_analysis_task, target_series, n_periods
        )

        # 4. 結果の返却
        history_dates = target_series.index.strftime('%Y-%m-%d').tolist()
        history_values = target_series.values.tolist()

        return {
            "status": "success",
            "metadata": {
                "stats_id": stats_data_id,
                "cat": cat,
                "area": area,
                "model_order": str(fit_result["order"]),
                "seasonal_order": str(fit_result["seasonal_order"]),
                "aic": fit_result["aic"],
                "is_white_noise": diagnosis["is_white_noise"],
                "lb_pvalue": diagnosis["lb_pvalue"]
            },
            "history": {
                "index": history_dates,
                "values": history_values
            },
            "forecast": forecast
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Analysis Error: {str(e)}")