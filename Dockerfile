FROM python:3.8-slim

# 必要なビルドツールと依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3-dev \
    libhdf5-dev \
    && pip install --no-cache-dir pybind11 h5py \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ作成
WORKDIR /app

# プロジェクト一式をコピー
COPY src/*.hpp src/
COPY src/*.cpp src/
COPY Makefile .
COPY setup.py .
COPY sisap2025.py .

RUN python3 -m pip install h5py pybind11

# Python拡張モジュールのビルド
RUN make

# For SISAP2025
ENV OMP_NUM_THREADS=8

# コンテナ起動時にPythonを対話的に使えるようにする
CMD ["bash"]
