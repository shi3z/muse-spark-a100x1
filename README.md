# Muse-Spark-Open-816B-A42B を A100 80GB 一枚で動かす

[meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)
（816B 総パラメータ、42B アクティブ、256 expert top-8、62 層）を **A100 80GB 一枚 + ホスト RAM** で動かし、
素のデコードと投機デコード（DFlash2、および自前実装の DSpark）を比較した記録です。
[Strata](https://github.com/Niko1221/Strata) と同じ考え方（dense 重みは GPU、expert はホスト RAM で CPU 計算）を
llama.cpp master の `--n-cpu-moe` で実現しています。

詳細な英語レポートは [REPORT.md](REPORT.md)。

## 結果（GPU 2 占有、Xeon 4410Y x2 / 24 スレッド、Q4_K_S、128 トークン生成、greedy）

| モード | プロンプト | prefill tok/s | decode tok/s | draft 採択 |
|---|---|---|---|---|
| base | 158 tok | 12.1 | 5.67 | – |
| base | 158 tok (code) | 9.6 | 6.50 | – |
| base | 3009 tok | 39.6 | 5.96 | – |
| DFlash2 (n_max 7) | 158 tok | 11.7 | 5.62 | 61%、5.25 tok/cycle |
| DFlash2 (n_max 7) | 158 tok (code) | 8.6 | 3.18 | 27% |
| DFlash2 (n_max 7) | 3009 tok | 39.2 | 5.54 | 53% |
| DSpark 自前実装 (n_max 7, p_min 0.5) | 158 tok | 11.7 | 4.16 | 37%、2.22 tok/cycle |
| DSpark 自前実装 (n_max 7, p_min 0.5) | 158 tok (code) | 8.5 | 5.01 | 51%、1.73 tok/cycle |
| DSpark 自前実装 (n_max 7, p_min 0.5) | 3009 tok | 38.6 | 5.03 | 51%、2.12 tok/cycle |

3 モードとも生成テキストは一致（投機デコードはロスレス）。

### わかったこと
1. **動く。** GPU 側は約 24 GB（attention / dense FFN / embedding / lm_head、16K ctx）、expert 422 GiB はホスト RAM。
   `--load-mode none` + `numactl --interleave=all` + `--numa distribute` + `kernel.numa_balancing=0` で
   重みが両ソケットに 217 GB / 217 GB に均等配置され、素のデコードで **5.5〜6.5 tok/s**。
   （mmap のまま、片ソケットに偏り、GPU を他プロセスと共有、の状態では 0.4〜2.8 tok/s だった。）
2. **expert が CPU にある構成では投機デコードが効かない。** DFlash2 は 1 検証ステップで平均 5.25 トークン採択されるのに
   速度は変わらず、コードでは逆に半減。検証ブロックの各トークンがそれぞれ 8 expert/層を起こすので、
   検証コストがブロック長にほぼ比例してしまう（GPU 常駐なら「重みを一回読んで k トークン」で得をするが、ここでは成り立たない）。
   速くするなら Strata 的な GPU hot-expert キャッシュか、CPU expert カーネル側（現状 expert 重み 14 GB/token ≒ 80 GB/s で、
   実測ホスト帯域 250 GB/s の 1/3）。
3. **自前 DSpark 実装は公式の採択率レンジに入った。** 公式カード（7 draft、temp 1.0）は採択 28〜42%、2.7〜3.9 tok/cycle。
   こちらは greedy で 37〜51%。DFlash2 も公式 5.37 tok/cycle（GSM8K）に対し 5.25。

## 自前 DSpark 実装（llama.cpp へのパッチ）
`llama.cpp-patches/` に、llama.cpp master `2851b3bcb`（2026-10-16）に対するパッチを置いています。
* `conversion/muse_spark.py`: `MuseSparkDSparkModel` コンバータ（HF safetensors → DFLASH arch GGUF）。
  gated attention output、重みなし post-norm、norm-preserving residual gate（alpha/beta を温度 0.3 の sigmoid で事前計算）、
  delta 形式の norm（+1）、qk scale 5.473 を attn_q_norm に畳み込み、d2t、Markov head、confidence head。
* `src/models/dflash.cpp`: Muse Spark バックボーン（`wqkv_gate` の有無で切替）。
* `src/models/muse-spark.cpp`: 最終層出力を target layer `n_layer` として公開（DSpark の target_layer_ids が 61 で終わるため）。
* `gguf-py/gguf/constants.py`, `conversion/__init__.py`: テンソル定義と登録。

適用: `git -C llama.cpp checkout 2851b3bcb && git am path/to/llama.cpp-patches/*.patch`、
ビルドは `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DGGML_NATIVE=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`。
変換: `python convert_hf_to_gguf.py Muse-Spark-Open-DSpark/ --outtype bf16`（`config.json` の `architectures` を `MuseSparkDSparkModel` に）。

## スクリプト
| ファイル | 役割 |
|---|---|
| `run_muse_spark.sh` | llama-server 起動（GPU/CTX/THREADS/NCMOE/PORT/LOAD_MODE/NUMACTL を env で指定） |
| `run_muse_cmp.sh MODE` | base / dflash / dspark の 3 構成 |
| `start_muse.sh MODE [log]` | tmux `muse` で再起動して起動完了まで待つ（tmux は呼び出し元の env を継承しないので明示的に渡す） |
| `compare_modes.sh` | 3 構成を順番に起動して `bench_client.py` で計測、`compare_results.log` に追記 |
| `bench_client.py PORT N` | 3 プロンプトで prefill/decode tok/s と draft 採択を表示 |
| `bench_threads.sh` | llama-bench によるスレッド数スイープ |
| `muse_*.log` | サーバーログ（`.badflags` / `.layer62abort` / `.rowsabort` は失敗した起動の記録） |

## 環境
2x Xeon Silver 4410Y（24 コア、AMX）、1.76 TB RAM、A100 80GB PCIe x8（うち 1 枚使用）、
llama.cpp master build 11675（CUDA sm80 + AMX）、重みは ggml-org の Q4_K_S GGUF（22 シャード、438 GiB）。
