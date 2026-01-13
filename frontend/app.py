import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Macro-Micro Linkage",
    page_icon="📈",
    layout="wide"
)

# --- 2. Constants & Settings ---
BACKEND_URL = "http://backend:8000"

STATS_CATALOG = {
    "消費者物価指数 (CPI) - 総合": {
        "id": "0003427113",
        "params": {
            "cdCat01": "0001", 
            "cdArea": "00000"
        },
        "desc": "【インフレ指標】物価の変動を表します。"
    }
}

# --- 3. Sidebar ---
st.sidebar.title("🎮 Control Panel")
st.sidebar.subheader("1. Select Indicator")
selected_name = st.sidebar.selectbox("分析対象の経済指標", options=list(STATS_CATALOG.keys()))
selected_meta = STATS_CATALOG[selected_name]

st.sidebar.subheader("2. Forecast Settings")
n_periods = st.sidebar.slider("予測期間 (Months)", 6, 36, 12)

# --- 4. Main Logic ---
st.title("📈 Macro-Micro Linkage Platform")
st.markdown(f"### {selected_name}")

if 'data' not in st.session_state:
    st.session_state['data'] = None

if st.button("🚀 Run AI Analysis", type="primary"):
    with st.spinner(f'Analyzing {selected_name}...'):
        try:
            req_params = selected_meta['params'].copy()
            req_params["n_periods"] = n_periods
            
            response = requests.get(
                f"{BACKEND_URL}/analysis/predict/{selected_meta['id']}", 
                params=req_params
            )
            
            if response.status_code != 200:
                st.error(f"Analysis Failed: {response.text}")
                st.stop()
            
            result_json = response.json()
            if not result_json.get("history") or not result_json.get("forecast"):
                st.error("Invalid Data")
                st.stop()
                
            st.session_state['data'] = result_json
            st.success("Analysis Complete!")

        except Exception as e:
            st.error(f"System Error: {e}")

# --- 5. Visualization & Simulation (Tabs) ---
if st.session_state['data']:
    data = st.session_state['data']
    
    # DataFrame化
    history_df = pd.DataFrame(data["history"])
    forecast_df = pd.DataFrame(data["forecast"])

    # カラム名のゆらぎ吸収
    rename_map = {}
    if 'index' in history_df.columns: rename_map['index'] = 'date'
    if 'values' in history_df.columns: rename_map['values'] = 'value'
    if rename_map: history_df = history_df.rename(columns=rename_map)
    if 'date' in history_df.columns:
        history_df['date'] = pd.to_datetime(history_df['date'])

    # Forecast側の処理
    if 'index' in forecast_df.columns:
        forecast_df['date'] = pd.to_datetime(forecast_df['index'])
        forecast_df = forecast_df.set_index('date')
    elif not isinstance(forecast_df.index, pd.DatetimeIndex):
        pass

    # タブ生成
    tab1, tab2 = st.tabs(["📊 Macro Forecast (未来予測)", "🎮 Business Simulator (経営判断)"])

    # === Tab 1: Macro Forecast ===
    with tab1:
        st.markdown(f"""
        #### 👁️ AI Analysis Report: {selected_name}
        過去のトレンドを学習したAIが、向こう**{n_periods}ヶ月**の推移を予測しました。
        """)
        
        # 重要な数値をKPIとして表示
        last_hist_val = history_df['value'].iloc[-1]
        last_pred_val = forecast_df['mean'].iloc[-1]
        change_rate = (last_pred_val - last_hist_val) / last_hist_val * 100
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("現在の値", f"{last_hist_val:.1f}")
        col_kpi2.metric("予測値 (期末)", f"{last_pred_val:.1f}", f"{change_rate:+.2f}%")
        col_kpi3.info("💡 **青い線**が予測シナリオ、**グレーの帯**は不確実性（リスク幅）を示します。")

        fig = go.Figure()

        # 描画用データ準備
        last_hist_date = history_df['date'].iloc[-1]
        last_hist_value = history_df['value'].iloc[-1]
        plot_forecast_df = forecast_df.copy()
        
        # 信頼区間
        x_ci = [last_hist_date] + list(plot_forecast_df.index) + list(plot_forecast_df.index)[::-1] + [last_hist_date]
        y_ci = [last_hist_value] + list(plot_forecast_df['upper']) + list(plot_forecast_df['lower'])[::-1] + [last_hist_value]
        
        fig.add_trace(go.Scatter(
            x=x_ci, y=y_ci,
            fill='toself', fillcolor='rgba(100, 100, 100, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval (リスク範囲)',
            hoverinfo="skip"
        ))

        # 実績
        fig.add_trace(go.Scatter(
            x=history_df['date'], y=history_df['value'],
            mode='lines', name='実績 (History)',
            line=dict(color='black', width=1.5)
        ))
        
        # 予測
        x_pred = [last_hist_date] + list(plot_forecast_df.index)
        y_pred = [last_hist_value] + list(plot_forecast_df['mean'])
        
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred,
            mode='lines', name='AI予測 (Forecast)',
            line=dict(color='blue', width=2)
        ))
        
        # 期間選択UI (Zoom機能)
        default_start = last_hist_date - pd.DateOffset(years=5)
        default_end = plot_forecast_df.index[-1] + pd.DateOffset(months=1)

        fig.update_layout(
            height=500, hovermode="x unified", template="simple_white",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            yaxis_title="Index Value", 
            xaxis=dict(
                title="Year",
                range=[default_start, default_end], # 初期表示は直近5年+未来
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(count=10, label="10y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Tab 2: Business Simulator ===
    with tab2:
        st.markdown("#### 📉 Cost Impact Simulator")
        st.markdown("""
        **「マクロ経済の変動は、自社の利益をどれくらい削るのか？」** AIが予測した物価上昇率をもとに、あなたのビジネスの**インフレ耐久力（生存可能性）**を診断します。
        """)
        
        if not history_df.empty and not forecast_df.empty:
            # 変動率計算
            current_val = history_df['value'].iloc[-1]
            future_val = forecast_df['mean'].iloc[-1]
            macro_change_pct = (future_val - current_val) / current_val * 100
            
            st.divider() # 区切り線

            col_sim_input, col_sim_viz = st.columns([1, 1.5])
            
            with col_sim_input:
                st.subheader("1. Parameters (設定)")
                
                # 自動入力項目
                st.info(f"📊 AI予測による物価上昇率: **{macro_change_pct:+.2f}%**")
                
                # ユーザー入力
                revenue = st.number_input("現在の年商 (百万円)", value=100.0)
                cost_ratio = st.slider("現在の原価率 (%)", 10, 90, 60)
                
                st.write("---")
                st.markdown("**👇 インフレ感応度 (重要)**")
                st.caption("世の中の物価が1%上がった時、あなたの仕入れ値は何%上がりますか？")
                sensitivity = st.slider(
                    "感応度 (1.0 = 物価と同じだけ上がる)", 0.0, 2.0, 1.0, step=0.1
                )
                
                st.markdown("**👇 アクションプラン**")
                price_hike = st.slider("販売価格の値上げ (%)", 0.0, 10.0, 0.0, step=0.1)

            with col_sim_viz:
                st.subheader("2. Simulation Result (結果)")

                # 計算ロジック
                current_cost = revenue * (cost_ratio / 100)
                current_profit = revenue - current_cost

                # 予測のみ（放置）
                # 物価上昇率 × 感応度 ＝ 実質コスト増
                cost_increase_pct = macro_change_pct * sensitivity
                cost_increase_factor = 1 + (cost_increase_pct / 100)
                
                future_cost_passive = current_cost * cost_increase_factor
                future_profit_passive = revenue - future_cost_passive
                
                # 対策後
                future_revenue_active = revenue * (1 + price_hike / 100)
                future_profit_active = future_revenue_active - future_cost_passive
                
                # チャート描画
                fig_sim = go.Figure()
                x_vals = ["現在", "放置した場合", "値上げ対策後"]
                y_vals = [current_profit, future_profit_passive, future_profit_active]
                
                colors = ['gray', 'crimson' if future_profit_passive < 0 else 'salmon', '#00CC96']
                
                fig_sim.add_trace(go.Bar(
                    x=x_vals, y=y_vals,
                    marker_color=colors,
                    text=[f"{v:.1f}百万円" for v in y_vals],
                    textposition='auto',
                ))
                
                fig_sim.update_layout(
                    title="営業利益の推移予測",
                    yaxis_title="営業利益 (百万円)",
                    height=350,
                    template="plotly_white"
                )
                st.plotly_chart(fig_sim, use_container_width=True)
                
                # 診断コメント
                st.markdown("##### 📝 Diagnosis")
                if future_profit_passive < 0:
                    st.error(f"⚠️ **危険:** このままでは原価高騰により**赤字転落**します。少なくとも **{(abs(future_profit_passive)/revenue*100):.1f}%** 以上の値上げが必要です。")
                elif future_profit_passive < current_profit * 0.9:
                    st.warning(f"⚠️ **注意:** 利益が減少します。コスト増（+{cost_increase_pct:.1f}%）を吸収するための対策を検討してください。")
                else:
                    st.success("✅ **健全:** 現在のコスト構造なら、インフレ影響を吸収して利益を維持可能です。")

else:
    st.info("👈 サイドバーから分析を開始してください")