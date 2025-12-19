# データセットの生成
```bash
docker compose run --rm synth_misinfo
```
- 生成するデータセットの種類を変更したい場合は，`docker-compose.yaml`の`synth_misinfo`内にある`command`の第８引数を{`easy`, `medium`, `hard`}のいずれかにする．
- あらかじめルートディレクトリに`/outoputs`を作成しておく必要がある．

# OCSVMの実行
```bash
docker compose run --rm train_ocsvm
```