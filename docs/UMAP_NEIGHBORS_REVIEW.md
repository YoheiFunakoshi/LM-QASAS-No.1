# UMAPの近傍数を一つずつ比較する

この補助解析は、共同研究者から原作図条件の回答を待つ間に、現行の図がUMAP設定でどの程度変わるかを調べるものです。通常のアプリの設定や候補順位は変更しません。入力条件の比較は保留し、[AbLang2平均方法の比較](POOLING_REVIEW.md)に続いて、現行の公式seqcodingの数値表現を固定します。

## 1. 何を変えるか

`n_neighbors`は、UMAPが各点の周辺をどの程度広く見て配置を作るかを指定します。小さい値は局所的な関係、大きい値はより広い範囲の関係を重視する傾向があり、細かな構造との兼ね合いがあります。[UMAP公式の説明](https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors)

今回の値は実行前に15・30・50と定め、それぞれを同じ2つのseedで計算します。15は現行値、30と50は段階的に広げた検討用の値です。論文で使われた値が確認されたという意味ではありません。論文の候補探索にあるk-NNのkと、UMAPの`n_neighbors`は別の設定です。

| 項目 | 今回の扱い |
| --- | --- |
| UMAP近傍数 | 15・30・50 |
| 乱数seed | 20260919・20260920を各条件に対応させる |
| 入力 | 現行のD不使用、read数1を含む採用集合、V/J/CDR-H3/isotypeのclone定義を保持 |
| embedding | 保存済みの公式AbLang2 seqcoding。再推論しない |
| UMAPへ渡す点 | 3時点の全採用clone観測。Top Nで制限しない |
| その他のUMAP設定 | cosine、min_dist=0.1、spread=1、200 epochs、random初期化等を固定 |
| 重複CDR-H3 | embeddingを再利用するが、clone観測は保持する |
| Counts | UMAPやKDEの重みに使わない |
| 候補順位・赤点 | 今回の比較対象外 |

15の保存座標は、入力・embedding・clone順・全設定・実行ライブラリが一致することを確認して再利用します。計画、座標、設定、hash、コードの由来を新しい結果フォルダーへ保存します。

## 2. 図の比較条件

独立したUMAPは回転・鏡映・平行移動で向きが変わり得るため、対応する全cloneを使って現行の基準座標へ向きを合わせます。拡大縮小や論文画像への形合わせは行いません。変換前の座標も保存します。

計算した6条件で、KDEの帯域幅の数値を現行基準の値に固定し、格子・軸・色尺度を共通化します。各時点の密度積分を共通格子上で1にそろえます。値はclone総数を表しません。同じ帯域幅でもUMAPの広がりが変われば相対的な平滑化の強さは変わり得るため、この比較だけでKDEの妥当性を確定しません。

2枚の図を作り、それぞれ「論文原図、15、30、50」の順に表示します。両seedを同じ条件で示し、見た目が良いseedだけを採用しません。論文原図は提供された画像を保持し、計算図とは座標と色尺度が別であることを明記します。原図の座標がないため、論文との座標一致度は計算しません。

## 3. 見た目に加えて確認すること

入力の480次元で近かった配列が、2次元でも近くに置かれるかを補助的に調べます。同じCDR-H3のコピーを近くに置くだけで評価が高くなることを避けるため、**この診断だけ**は各CDR-H3の全clone座標の重心を使います。UMAPやKDEに使う点は統合しません。

- 長さ・保存indexの順序から決定的に512種類までを選び、全条件で同じqueryを使います。母集団からの無作為標本ではありません。
- 探索相手は全ての異なるCDR-H3です。自分自身を除外し、入力はcosine距離、2次元重心はEuclidean距離で厳密探索します。
- 評価する近傍数は15と50の両方を全条件に共通で使います。UMAPに指定する値に合わせて評価基準を変えません。
- 共通する近傍の割合をqueryごとに保存します。同距離は保存index順で扱い、境界の近似同距離件数も残します。
- 重心が同配列の点の散らばりを隠さないか、重心まわりのRMSと「RMS÷第k近傍までの距離」を確認します。
- seedによる全体配置の違いは、時点別に選ぶ固定clone標本の点間距離順位相関と、全cloneを剛体整合したRMSで補助的に示します。

近傍一致率は2次元化による関係の保持を示す限定的な診断です。抗原特異性、候補抽出性能、論文図の復元を証明する値ではありません。2つのseedだけで乱数に対する安定性を保証しません。

## 4. 実行する

プロジェクト専用のPython環境を使用します。`SOURCE_RUN`、`SOURCE_PROJECTION`、`POOLING_COMPARISON`は保存済みの対応する結果、`NEW_COMPARISON`は存在しない新しい保存先に置き換えてください。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) 'local_records\numba_cache')
.\.venv\Scripts\python.exe scripts\compute_neighbors_comparison.py --run-dir SOURCE_RUN --projection-dir SOURCE_PROJECTION --pooling-dir POOLING_COMPARISON --output-dir NEW_COMPARISON
.\.venv\Scripts\python.exe scripts\render_neighbors_comparison.py --comparison-dir NEW_COMPARISON --reference-image LOCAL_REFERENCE_IMAGE
```

補助コードは同じ`scripts/`内のpooling比較コードを使用します。計算はローカルで行い、ネットワークを使う推論は行いません。原本や旧結果を上書きせず、既存の出力先は拒否します。中断したフォルダーを完成済みとして再利用せず、新しい保存先で再実行します。

主な出力は`experiment_plan.json`、各条件のraw/aligned座標と密度格子、`neighborhood_diagnostics.npz`、`comparison_metadata.json`、各seedの比較PNG、`render_metadata.json`です。いずれも実データ由来の場合は公開GitHubへ登録しません。計算完了のmetadataと、作図・検証の完了を分けて確認してください。

## 5. 判断と検証の範囲

見た目の変化、近傍関係の保持、seed差を併せて確認し、近傍数だけで論文図との差を説明できるかを考えます。標準設定へ採用する判断は比較の実施と分けます。次の別要因を試す場合も、現行条件を起点として一つずつ変えます。

検証では人工データの対応・設定固定・保護機能と、実成果物の入力hash、全座標、数値診断、共通密度条件を確認します。Gaussian密度の抜取検算は、極小の裾で相対誤差だけが大きく見えることを考慮し、許容値を`1.05e-6 × 厳密値 + 1e-12 × パネル最大密度`と事前に定めます。全格子での厳密精度保証やUMAP再実行による独立再現とは区別します。実施結果は[開発記録](DEVELOPMENT_LOG.md)と非公開の日本語結果記録に残します。
