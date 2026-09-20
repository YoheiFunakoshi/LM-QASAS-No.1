# UMAPの初期配置と距離尺度を別々に比較する

## この比較で確かめること

[min_dist](UMAP_MINDIST_REVIEW.md)と[KDE帯域幅](KDE_BANDWIDTH_REVIEW.md)に続き、点を置き始める方法と、元の480次元で近さを測る方法を切り分けます。見た目に合う組合せを探索して採用するのではなく、どの要因が図と近傍関係に影響するかを確認します。

| 条件名 | 初期配置 init | 距離 metric | 現行からの変更 |
| --- | --- | --- | --- |
| baseline | random | cosine | なし |
| spectral | spectral | cosine | 初期配置のみ |
| euclidean | random | euclidean | 距離尺度のみ |

3条件をseed=20260919/20260920の両方で比較します。spectralとEuclideanの同時変更は含みません。現行基準2fitは元入力・embedding・全設定・hashを照合して再利用し、4fitを新しく計算します。

`random`は乱数による開始位置、`spectral`は近傍グラフを使った開始位置です。`cosine`はベクトルの方向、`Euclidean`はベクトル間の直線距離を基準にするため、後者では大きさの差も関係します。Euclidean比較の前に単位長への正規化やスケーリングを追加しません。[UMAP公式API](https://umap-learn.readthedocs.io/en/latest/api.html)

UMAPの公式既定値にspectral/Euclideanが含まれることは、論文がそれを使った証拠ではありません。今回も原設定は要確認です。同じseedを対応させても、初期化で乱数の消費が変わるため、その後の最適化まで同じ乱数列になるとは限りません。

## 固定するものと保存するもの

入力集合、clone定義、公式AbLang2の保存embedding、n_neighbors=15、min_dist=0.1、spread=1、200 epochs、その他の設定を固定します。全clone観測をUMAPとKDEへ渡し、Top Nで絞りません。Countsで重み付けせず、候補抽出は実行しません。入力・標準運用は変更しません。

6条件を現行基準に回転・鏡映・平行移動で合わせ、拡大縮小はしません。現行基準のKDE帯域幅の数値を固定し、同じ格子・軸・色尺度で表示します。密度は各時点の有限格子上の積分を1とします。固定帯域幅でも配置の広がりが変われば相対的な平滑化は変わるため、この影響は残ります。

各seedについて「論文原図・現行・spectral」と「論文原図・現行・Euclidean」の図を作り、合計4枚を保存します。論文画像へ形や色を合わせる変形をせず、原図と計算図の座標・色尺度が別であることを明示します。赤点は評価しません。

## 距離を変える比較の公平な診断

全条件について、480次元のcosine近傍とEuclidean近傍の**両方**に対する一致率を計算します。一方だけでEuclideanの良否を判定しません。元の480次元でも両距離の近傍がどれだけ一致するかを併記し、近さの定義の変更と2次元化による変化を区別します。

同じCDR-H3の全clone座標を重心へまとめるのは診断時だけです。同じ決定的なquery標本（最大512種類）に対し、全CDR-H3候補から自己点を除いて近傍15/50種類を厳密探索します。同距離は保存index順で処理し、境界の近似同距離件数も残します。入力はUMAPと同じfloat32値を、距離の検算ではfloat64へ変換します。

重心によって同一CDR-H3のcloneの散らばりが隠れるため、重心からのRMSと近傍半径との比も記録します。さらに固定したclone標本の点間距離順位相関、全cloneの剛体整列後RMSでseed差を確認します。これらは記述的な診断で、生物学的性能、正式なtrustworthiness、論文図の再現率ではありません。2seedのみで最適性や再現性を保証しません。

## spectralが実際に動いたか

各新規fitの警告を保存し、spectralの固有値計算が失敗してrandomへ切り替わった場合は、既定では停止します。失敗の警告記録は保持します。

失敗を調べ、他の独立比較も続ける必要がある場合に限り、新しい出力先へ明示的に`--retain-spectral-fallback`を指定できます。この場合、fallbackがあれば全体を`completed_with_fallback`とし、該当seedの図へ「参考：spectral指定（randomへの切替を検出）」と表示します。警告を消して正常成功に見せる機能ではありません。これはspectralの純粋な比較としての成功を意味せず、設定変更の採用根拠には使いません。独立に計算したEuclidean比較と区別します。

導入済みUMAPの実装では、小さい非連結成分に警告なしでrandomの局所配置を使うこともあります。固定200 epochsに対応する辺の剪定（重みが最大値/epochs未満の辺を除去）後の連結成分数・サイズ、小さい成分の点数を保存します。「solver fallbackの警告なし」と「全点がspectralで局所初期化された」は区別します。既存random基準では当時のグラフ・警告は保存されていないため未取得と記録します。

## 再実行と成果物

プロジェクト専用Pythonを使用します。以下の名前は説明用で、ローカルの対応するパスへ置き換えます。出力先には既存結果と異なる新しいフォルダーを指定します。

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) 'local_records\numba_cache')
.\.venv\Scripts\python.exe scripts\compute_init_metric_comparison.py --run-dir SOURCE_RUN --projection-dir SOURCE_PROJECTION --pooling-dir POOLING_COMPARISON --output-dir NEW_COMPARISON
.\.venv\Scripts\python.exe scripts\render_init_metric_comparison.py --comparison-dir NEW_COMPARISON --reference-image LOCAL_REFERENCE_IMAGE
```

AbLang2の再推論やモデルダウンロードは不要です。計画、全設定、由来hash、警告・連結成分、raw/aligned座標、密度、両参照距離の診断、作図条件を保存します。原本・旧成果物は読み取りだけとし、実データ由来の図・値・詳細記録はGit管理外に置きます。

標準設定への採用と比較実施は別の判断です。今回の検証範囲は[開発記録](DEVELOPMENT_LOG.md)、詳細な結果と図は非公開の日本語結果記録で確認します。共同研究者のコード・設定が得られたら、今回の感度分析とは区別して照合します。警告文だけから、どの連結成分が失敗したか、入力重複が原因だったかを断定しません。
