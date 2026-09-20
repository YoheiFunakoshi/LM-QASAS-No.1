# UMAPのmin_distによる配置の違いを比較する

## 目的と今回の条件

[近傍数の比較](UMAP_NEIGHBORS_REVIEW.md)に続き、現行の入力・公式AbLang2の保存embedding・近傍数15を固定して、`min_dist`だけを変えます。`min_dist`は2次元で点をどの程度密集させるかに関係する設定です。[UMAP公式の説明](https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist)。個々の点の距離が必ずその値以上になるという、出力に対する厳密な距離制約とは扱いません。

今回の比較は0.1（現行）・0.3・0.5を事前に定め、それぞれを同じ2seedで計算します。論文の設定を特定したものではありません。数値が大きいほどよいという判断や、標準値の自動変更は行いません。

| 条件 | 今回の扱い |
| --- | --- |
| min_dist | 0.1・0.3・0.5を比較 |
| n_neighbors | 現行15で固定 |
| 入力集合・AbLang2 | D不使用・read数1を含む現行clone集合と公式seqcodingを固定 |
| UMAP観測点 | 全採用clone。Top Nによる制限なし |
| その他のUMAP条件 | cosine、random初期化、spread=1、200 epochs等を固定 |
| 乱数seed | 20260919・20260920を各条件に対応 |
| KDE | 現行基準の帯域幅数値を固定。全6条件共通の格子・軸・色尺度 |
| 候補選択・赤点 | 比較対象外。候補順位は変更しない |

## 図と数値診断の読み方

対応する全cloneを使い、各結果を現行基準へ回転・鏡映・平行移動で合わせます。拡大縮小や論文画像への形合わせはしません。変換前の座標を保存し、論文原図とは座標・色尺度が別であることを図に明示します。現行の0.1は入力・設定・保存hashを照合して既存座標を再利用します。

各seedについて、論文原図と0.1・0.3・0.5を並べた図を1枚ずつ作成します。両方のseedを示し、良く見える方だけを選びません。密度は各時点の格子上の積分を1とし、Countsを重みに使いません。固定帯域幅でも点の広がりが変われば相対的な平滑化の強さは変わるため、KDEの影響とは区別して解釈します。

近傍保持の診断は前段階と同じです。同一CDR-H3の全clone座標の重心を**診断だけ**に用い、決定的に選ぶ同じ512種類までのqueryについて、全CDR-H3候補から厳密に探す近傍15・50種類の一致率を求めます。重複cloneの重心まわりの散らばり、近傍境界の同距離、seed間の配置差も残します。詳細と限界は[近傍数比較の診断](UMAP_NEIGHBORS_REVIEW.md#3-見た目に加えて確認すること)を参照してください。

今回の近傍一致率は、生物学的性能、正式なtrustworthiness、論文図の再現率ではありません。原図の座標・元設定がないため、画像が似たことだけで元法と同じとは判断しません。

## 再実行

プロジェクト専用Pythonを使います。以下の保存先・入力名は説明用で、対応するローカルパスへ置き換えてください。既存結果を上書きせず、新しい出力先を指定します。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) 'local_records\numba_cache')
.\.venv\Scripts\python.exe scripts\compute_mindist_comparison.py --run-dir SOURCE_RUN --projection-dir SOURCE_PROJECTION --pooling-dir POOLING_COMPARISON --output-dir NEW_COMPARISON
.\.venv\Scripts\python.exe scripts\render_mindist_comparison.py --comparison-dir NEW_COMPARISON --reference-image LOCAL_REFERENCE_IMAGE
```

同じ`scripts/`内のpooling・neighbors比較の検証済み補助関数を使用します。モデルの再学習・再推論は不要です。計画・全設定・元入力と成果物のhash・各条件のraw/aligned座標・密度・診断・作図条件を保存します。原本と旧結果は読み取りのみとします。実データ由来の出力と詳細結果はGit管理外に置きます。

## 判断と残る候補

見た目、近傍関係の保持、seed差を併せて評価し、標準設定へ反映する判断と感度分析の実施を分けます。検証結果と今回の判断は[開発記録](DEVELOPMENT_LOG.md)と非公開の日本語結果記録に残します。

この後、別実験として[固定座標でのKDE帯域幅](KDE_BANDWIDTH_REVIEW.md)を比較しました。残る候補はUMAPの初期配置（randomとspectral）、距離尺度（cosineとEuclidean）です。初期配置は別の開始状態から同じ関係を最適化する比較、距離尺度は何を近いとするか自体を変える比較なので同時に変更しません。これらは本段階のmin_dist比較では実施していません。入力条件の変更は引き続き保留し、共同研究者の回答と区別します。
