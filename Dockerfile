# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# （将来的にLightGBM学習まで同一コンテナでやる想定なら）OpenMPランタイム
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係
COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

# ソース一式（スクショの構造に合わせてコピー）
COPY date/ /app/date/
COPY src/ /app/src/

# デフォルトはデータ生成を実行（必要なら docker-compose の command で上書き可能）
ENTRYPOINT ["python", "/app/date/makeText.py"]
