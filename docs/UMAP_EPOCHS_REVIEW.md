# UMAPの最適化回数を比較する

## 目的と条件

[初期配置・距離尺度の比較](UMAP_INIT_METRIC_REVIEW.md)に続き、現行のrandom/cosineで `n_epochs` を200・500・1,000に変え、2次元配置への影響を調べます。AbLang2の保存ベクトルは共通です。回数を増やせば論文図へ近づく、または候補抽出が改善する、と仮定しません。

| 項目 | 扱い |
| --- | --- |
| n_epochs | 200（現行）・500・1,000 |
| seed | 20260919・20260920を全条件に対応 |
| 観測点 | 現行の全採用clone。Top Nで絞らず、Countsで重み付けしない |
| 数値表現 | 保存済みの公式AbLang2 seqcoding、同じfloat32入力 |
| 他のUMAP設定 | random、cosine、n_neighbors=15、min_dist=0.1、spread=1等を固定 |
| 候補抽出・入力条件 | 変更しない。赤点の比較は対象外 |

基準200回の2fitは入力・embedding・全設定・hashを検証して再利用します。500回と1,000回は、各seedで最初から計算する4fitを追加します。途中の200回の座標を出発点にしません。公式APIは `n_epochs` を低次元配置の最適化回数として説明しています。[UMAP公式API](https://umap-learn.readthedocs.io/en/latest/api.html)

## 「回数だけを変える」の意味と限界

変更する指定パラメータは `n_epochs` です。ただし導入済みUMAPでは、この値が最適化の長さ以外にも影響します。

- 弱い辺の除外閾値は「最大辺重み / n_epochs」。大きい回数では弱い辺が残る場合があります。
- 学習率の減衰予定も総回数に依存します。初期値を固定しても、同じ途中回数での動きの大きさは変わります。

したがって、同一の200回の計算を単純に延長した比較や、最適化対象の辺まで完全固定した比較ではありません。`n_epochs=[200,500,1000]` のような途中保存も、最大回数に対応する予定で動くため今回の独立した設定比較とは異なります。元の実装は `umap/umap_.py` の `simplicial_set_embedding` と `umap/layouts.py` の `optimize_layout_euclidean` で確認し、実行環境の版はmetadataに保存します。

図の変化や近傍一致率が良くなったことだけで、収束や「元の計算不足が原因」を証明したとは扱いません。各新規fitの警告と、指定回数に対応する剪定後の連結成分を記録します。以前のspectral比較を引き継いだ結果とは混同しません。

## 図と診断

各seedについて、論文原図・200回・500回・1,000回の4段の図を作り、両方を保存します。拡大縮小をせず、全clone対応で回転・鏡映・平行移動を合わせます。論文の座標と色尺度は別のままです。

KDE帯域幅は現行基準の数値を固定します。全6条件の座標を含む共通範囲に、各軸の余白 `max(3h, 0.05×幅)` を加えます。格子は各軸最低96点、間隔が `h/2` 以下となる共通サイズにします。必要サイズが512点を超える場合は、点や表示範囲を削らずに停止します。6条件で格子・軸・色を共通化し、各時点の共通有限格子上の密度積分を1とします。配置が広がると相対的な平滑化の強さは変わるので、固定帯域幅であることも解釈に含めます。

前段階と同じ診断方法を再利用し、同一CDR-H3の全clone重心を**診断時のみ**使います。同じ最大512queryを配列長・保存index順の分位点から決定的に選びます。無作為な代表標本とは扱いません。全CDR-H3を候補に、自己点を除いてcosine・Euclidean両方の480次元近傍と2次元近傍を照合します。k=15/50の一致率、境界同距離、重複cloneの散らばり、seed間の距離順位相関などを保存します。生物学的性能や論文再現率とは呼びません。

## 再実行と保存

プロジェクト専用Pythonを使用し、以下の説明用パスをローカルの対応する保存先へ置き換えます。既存結果と異なる、新しい出力先を指定します。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) 'local_records\numba_cache')
.\.venv\Scripts\python.exe scripts\compute_epochs_comparison.py --run-dir SOURCE_RUN --projection-dir SOURCE_PROJECTION --pooling-dir POOLING_COMPARISON --output-dir NEW_COMPARISON
.\.venv\Scripts\python.exe scripts\render_epochs_comparison.py --comparison-dir NEW_COMPARISON --reference-image LOCAL_REFERENCE_IMAGE
```

計画、全設定、元入力と成果物のhash、raw/aligned座標、密度、診断、警告と作図記録を保存します。共通の検証済み補助関数を再利用します。元入力・旧結果は読み取り専用で、実図・個人別の値・詳細監査はGit管理外です。通常のアプリや標準回数の変更は、この比較実施とは別に判断します。

検証の実施範囲は[開発記録](DEVELOPMENT_LOG.md)、詳細結果と図は非公開の日本語レポートで確認します。論文の回数設定は未確認のままです。
