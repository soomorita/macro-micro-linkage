FROM python:3.11-slim

WORKDIR /app

# 1. ビルドに必要な最小限のOSパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. pip自体のアップグレード（安定性向上）
RUN pip install --no-cache-dir --upgrade pip

# 3. 依存関係のインストール
# --default-timeout=1000 を付与し、巨大なライブラリのDL失敗を防ぐ
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. ソースコードのコピー
COPY . .

# 5. ポート開放
EXPOSE 8000

# 6. 【最重要】サーバー起動コマンド
# コンテナ起動時に自動的にFastAPIを立ち上げる
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]