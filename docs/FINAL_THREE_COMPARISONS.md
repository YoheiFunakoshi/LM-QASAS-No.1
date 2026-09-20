# 最後の3項目に限定した比較

2026-09-20。利用者は「PCAなどの前処理」「別の密度推定方法」「UMAPへ渡す配列集合」の3項目を追加し、その後いったん停止する方針を指定した。本書は条件と再実行方法の公開用記録。実配列・研究画像・個別の数値・検証結果はGit管理外に保存する。

## 比較条件を先に固定する

| 項目 | 今回だけ比較する条件 | 固定するもの・読み方 |
| --- | --- | --- |
| UMAP前の処理 | 現行480次元／全clone平均を引くだけ／同じ平均を用いるPCAで累積寄与率95%以上を保持 | 同じ採用clone、AbLang2、UMAP設定と等方KDE。中心化と次元削減を分ける |
| 密度の推定 | 現行等方Gaussian／pooled Scott則の共分散型Gaussian | 同じ現行UMAP座標。3時点には共通の帯域行列を使う。幅の規則と方向性の両方が変わる |
| UMAPの計算単位 | 全clone観測／同じCDR-H3を一度だけUMAPへ渡し、全cloneへ座標を戻す | 採用配列、clone定義、元表現とKDEを固定。表示上は両方式とも全cloneを保持 |

**各条件を同じ2seedで比較し、別条件の追加・組合せ探索へ進まない。** 標準UMAPはrandom、cosine、n_neighbors=15、min_dist=0.1、n_epochs=200等を保存設定から照合して使う。現行の両seedの座標は以前の保存成果物を検証して再利用する。中心化・PCA・CDR-H3一意化の各2fitが新規計算に当たる。

利用者は、配列集合について「採用配列を固定し、UMAPでの重複の扱いだけ比較する」と回答した。D不使用、read数1を含める条件、isotypeを含む現在のclone定義を維持する。モデル推論、候補スコア、Top N選択をやり直す比較ではない。赤点は今回の検討に含めない。

## 1. PCAの意味と対照条件

保存された元表現をUMAPに合わせてfloat32とし、PCA計算ではfloat64にして全clone観測を3時点合同で使用する。重複cloneはこの計算で保持する。`PCA(n_components=None, svd_solver='full', whiten=False)` を一度だけfitし、成分と平均・分散を保存する。

累積寄与率が95%以上となる最小次元を、結果の図を見る前に `searchsorted(..., side='left')` で決める。中心化だけの対照も、同じfitから得た平均を用いる。個々の時点で別々のPCAは行わない。UMAP入力は各条件ともfloat32へ変換する。[scikit-learn PCA仕様](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

95%は今回の事前に決めた感度分析条件であり、論文の設定とは未確認。中心化はcosine距離を変え得るため、元表現と前処理後の高次元近傍の一致、それぞれに対する2次元の近傍保持を別々に記録する。保持分散は抗原特異性や近傍保持の保証ではない。

## 2. 方向性も考慮する密度推定

比較する共分散型KDEでは、各seedの全clone座標から標本共分散S（ddof=1）を求める。2次元Scott則の係数 f=N^(-1/6) を用い、帯域行列 H=f²S を作る。**そのseedの3時点に同じHを使用する。** 異なるseedの座標は混ぜない。

これはSciPy公式のScott則と共分散の定義に従う実装であり、時点ごとに独立して `scipy.stats.gaussian_kde` をfitする方法とは異なる。[SciPy公式仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)

計算はCholesky変換後のGaussian KDEと密度のJacobian補正を用いる。`KernelDensity`はball_tree、rtol=1e-6、atol=0。現行の等方型は保存済みの数値帯域幅を保持する。両方式とも各cloneを等重みとし、各時点の共通有限格子上の積分を1にする。幅の規則と方向性を同時に変える比較なので、差を方向性だけの効果と断定しない。

## 3. CDR-H3の重複と図の観測数

一意化するのはUMAPのfit時だけ。同じCDR-H3のIDを保存順で一度ずつ渡し、得られた座標を保存されたembedding indexで元の全clone行へ戻す。異なるCDR-H3のベクトル値が偶然同じでも、数値の一致だけで別配列を統合しない。

この変更は同一配列の重複度と近傍グラフに影響する。全cloneを背景・密度推定へ戻すため、clone単位の時点別観測は保持される。inverse map後に同じCDR-H3の点が完全に重なるのは構造上の結果であり、投影品質が改善した証拠にはしない。採用行・clone定義や最終Top Nの数え方は変えない。

## 図・数値診断・保存

- 比較項目ごとに、全条件と両seedを含む共通格子・軸・色尺度を使う。3項目の間では尺度が異なり得る。
- 最小カーネル標準偏差の半分以下となる格子間隔、少なくとも96点／軸、全座標と最大標準偏差3倍以上の余白を確保する。512点／軸を超える場合は、切り捨て・間引きをせず停止する。
- 座標の向き合わせは回転・鏡映・平行移動だけ。拡大縮小は行わない。論文の原図は独自の軸・色尺度であることを明示する。
- 近傍診断だけ同じCDR-H3のclone座標を重心にまとめる。配列長と保存順の分位点で固定した最大512queryから、全種類を候補に自己点を除き近傍15/50を厳密検索する。無作為標本ではない。
- 元表現・中心化後・PCA後の高次元cosine近傍をそれぞれ基準にする。2次元側はEuclidean。近傍の同距離は保存index順で決め、境界距離も保存する。
- seed間の座標診断は、各時点最大350cloneの固定標本で点間距離の順位相関を求める。拡大縮小をしないRMSも記録するが、配置全体の大きさに影響される。

成果物のhash、原本・元run・元projection・再利用する比較の不変を確認する。非有限値、cosineを定義できないゼロノルム、非正定値H、予期しないsolver fallbackは停止理由とし、黙って別条件へ置き換えない。失敗した成果物にも上書きしない。

## 実行する場所

プロジェクトの専用Pythonで、次の引数に手元の保存フォルダーを指定する。実配列・図・個別監査は公開しない。

```powershell
.\.venv\Scripts\python.exe scripts\compute_final_three_comparisons.py --run-dir '<元run>' --projection-dir '<元projection>' --pooling-dir '<検証済みpooling比較>' --output-dir '<新しい出力先>'
.\.venv\Scripts\python.exe scripts\render_final_comparisons.py --comparison-dir '<今回の出力先>' --reference-image '<提供された原図>'
```

計算はネットワークを遮断して保存済み表現を使う。`experiment_plan.json`を先に保存し、完了後の`comparison_metadata.json`、PCAモデル、条件別座標・密度、近傍診断を保存する。作図は保存された数値だけを読み、3項目×2seedの6図を作る。

新しいコードは`compute_final_three_comparisons.py`、`final_comparison_density.py`、`render_final_comparisons.py`。共有する出典・近傍・剛体整列・描画の検証処理は既存スクリプトを再利用する。

## ここで止める

今回の3項目の結果と限界を記録したら、一旦停止する。似た図が得られても、論文の原条件と断定したり、通常アプリへ自動採用したりしない。明確に似なくても、次の条件を自動追加しない。元条件と照合する際の再開方針は[停止・再開の判断規則](REPRODUCTION_STRATEGY.md)に従う。
