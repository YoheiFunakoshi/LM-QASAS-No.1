# 論文 Fig. 1c と現行アプリの比較方法

論文図とアプリの出力を並べるときは、背景の計算条件と、点が表す配列の選び方を分けて確認します。**似た見た目だけでは論文の再現成功を判定できません。** この文書は比較の手順と限界を説明します。個別の研究結果と比較画像はローカルだけに保存します。

Materials and Methodsとコードを項目ごとに照合した結果は、[マテメソ照合記録](METHODS_REVIEW.md)を参照してください。Dの注釈、cloneの細部、UMAPとk-NNの設定の区別、KDE・poolingを整理しています。

## 1. 何を比べているか

| 要素 | 論文 Fig. 1c | 現行アプリ |
| --- | --- | --- |
| 背景 | レパトアのembeddingをUMAPで表示し、KDEで分布を可視化 | 3時点をまとめた共通UMAPと、時点別の正規化KDE |
| 強調した点 | CoV-AbDab登録配列との類似性で関連するとした配列 | Peakで選んだTop N種類のCDR-H3と完全一致する各時点のclone観測 |
| 点の選択基準 | 論文では長さ正規化Levenshtein距離 ≤ 0.15 | K-meansのPeak増加スコア、CDR-H3の重複除去、Top N |
| 座標・色の数値 | 論文側のUMAP・KDE条件による | [可視化の解説](VISUALIZATION.md)に記録した暫定条件による |

論文の赤点は、その被験者の配列すべてについて抗原結合を実験確認したという意味ではありません。現行アプリの候補の印も抗原特異性の確認済みラベルではありません。両者の点数や位置を、そのまま一致率として扱いません。[論文本文・Fig. 1c・Methods 5.3–5.5](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2026.1844788/full)

Peak増加を基準として候補を選ぶので、候補がPeakに多く見えること自体は独立した精度検証ではありません。既知DBとの比較には同じDBの版・対象エントリ・絞り込み条件・距離の定義が必要です。

## 2. 同じにする必要がある条件

1. 入力集合と前処理：CDR-H3境界、機能注釈、曖昧なV/J候補、allele、isotype/subclass、clone定義。
2. embedding：AbLang2の版・重み・入力・poolingと、配列・clone・embedding行の対応。
3. 可視化：UMAPのmetric、n_neighbors、min_dist、初期化、seed、epochs、実行環境と、KDEの帯域幅・格子・正規化。
4. 候補を比較する場合：K-meansのK、初期化、ε、同点処理、候補数と重複除去の単位。
5. DB由来の点を比較する場合：CoV-AbDabの参照版、対象集合、照合規則または元の配列対応ラベル。

論文・ポスターに具体値がない条件は、現在の実装値を論文の値と断定しません。旧修士論文には異なる解析法もあるため、最新論文の未記載値を旧資料から自動的に補完しません。[資料対照](SOURCE_COMPARISON.md)

現在のExcel adapterはV/Jの機能ラベル `F` を要求しますが、Dの注釈・機能は採否に使いません。`x` を一律に「生物学的に非機能」と読み替えることはできません。共同研究者側の前処理との一致は未確認です。[Excel入力規則](TAKARA_INPUT.md)

[第1段階の入力集合確認](INPUT_SET_REVIEW.md)では、全採用クローンが背景へ渡ることを点検し、isotype粒度・複数V/J候補・最低read数などの未確認事項を分けて記録しました。

UMAPの島の位置・面積・密度は、そのまま元の480次元空間の量ではありません。別々に計算した図では座標系も一致しません。図を見ながら設定や座標を変形して似せても、再現性の証拠にはなりません。[UMAP公式FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html)、[再現性の説明](https://umap-learn.readthedocs.io/en/latest/reproducibility.html)

## 3. 保存済み結果から比較画像を作る

環境は[準備手順](SETUP.md)で作成済みとします。CLI専用の補助スクリプトです。下記のパスは説明用であり、実際の保存先に置き換えてください。

```powershell
.\.venv\Scripts\python.exe scripts\compare_reference_figure.py `
  --run-dir "outputs\run_EXAMPLE" `
  --projection-dir "outputs\run_EXAMPLE\projection_EXAMPLE" `
  --reference-image "local_records\reference\provided_fig1c.png" `
  --output-dir "outputs\figure_comparison_EXAMPLE"
```

- `--reference-image`：利用者が用意したFig. 1cの参照画像。PNGを推奨します。図の正しい出典と対象は事前に人が確認します。画像内容の自動照合はしません。
- `--selection-dir`：別の候補件数を比較したいとき、そのrunに属する選択フォルダーを指定します。省略時は初回の選択を使います。
- `--font-path`：日本語フォントのパス。既定はWindowsの `C:/Windows/Fonts/meiryo.ttc` です。他の環境では利用可能な日本語フォントを指定します。
- `--output-dir`：まだ存在しない新しいフォルダー。既存結果・原本・参照画像のフォルダーへの上書きを避けるため、既存の出力先は拒否します。

AbLang2・K-means・UMAP・KDEを再計算せず、保存済みの座標・密度と候補を読みます。参照画像の座標変形や、上下のUMAPの位置合わせも行いません。参照画像の赤点とアプリ候補の橙色の印を区別した2段の比較画像を出力します。

| 出力 | 内容 |
| --- | --- |
| `fig1c_comparison.png` | 上段が提供参照画像、下段が保存済み結果。上下の条件差を注記 |
| `comparison_metadata.json` | 元run・projection・selection・参照画像との対応、hash、強調したclone ID、検証範囲 |
| `.gitignore` | 出力のGit誤登録を防ぐ補助 |

元runとprojectionの対応、座標・密度両方のhash、clone数と座標行数、密度の有限値・非負値・積分1、候補の対応を検査します。読込対象の解析成果物・座標・密度・選択・参照画像は作図の前後でhashを照合します。この作図スクリプト単独では元Excel/CSVを再読込して全項目を再監査しません。

途中で失敗した場合、画像など一部の出力が残ることがあります。`comparison_metadata.json` の `status: completed` まで確認してください。再実行は新しい出力先を指定します。

## 4. 結論と記録の残し方

「図が作れた」「数値・対応の検査を通った」「論文と同じ条件だった」「候補の抗原特異性を検証した」は別の確認です。画像だけから候補の一致率や抽出性能を計算しません。元コードや座標がない間は、条件差と見えている相違を明記した比較にとどめます。

実画像・配列・検体名・個別集計・hash・提供資料は公開GitHubへ登録しません。比較画像は `outputs/`、詳細な監査と考察は `local_records/` に保存し、コード・一般的な手順・検証範囲だけをGitHubに残します。
