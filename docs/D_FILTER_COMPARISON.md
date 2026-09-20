# Dの注釈・機能を採否に使わない場合の見え方を比較する

**本書はD検査を行った旧policyのrunとの背景比較を説明します。** 現在のExcel新規解析は `takara-rg-hIGH20181210-v2-ignore-d` で、Dを採否に使いません。[現在の入力規則](TAKARA_INPUT.md)。旧runの候補数変更では新policyに移行せず、新規解析が必要です。

この補助解析は、旧Excel解析で使った条件から、**D遺伝子の注釈形式と機能ラベルによる除外だけを外す**と背景図がどう変わるかを確認する。利用者からDを無視するという見解を受け、この条件での比較を実施することにした。V/J、読み枠、CDR-H3の文字・長さ、isotype、cloneの定義、Countsを重みに使わない方針は維持する。

Dをclone keyに含めないことと、Dの注釈・機能で入力行を除外しないことは別である。元の実装でもDはclone keyに含めていない。今回の比較では入力のDに関する2つの除外条件を外す。`D=x`だけを許容する条件より広く、DのORF等の注釈も採否に使わない。全VDJの機能性を確認済みとする処理ではない。

最新論文5.3のfunctional V/D/Jという記載との対応は未確認のまま記録する。[論文Methodsとの照合](METHODS_REVIEW.md)。この比較スクリプトはアプリの標準入力条件を変更せず、論文を復元したという判定や、新しい候補の選択も行わない。

## 1. 計算するもの

1. 既存の完成run、原本、座標・密度のhashを照合する。現行adapterで全ブックの形式・集計値を確認し、元D情報から旧policyのclone集合・順序を復元して照合する。
2. 同じ原本を読み、Dに関する採否条件だけを外してclone集合を作る。既存の採用cloneがすべて新しい集合に含まれることを確認する。
3. 同じ配列の保存済みAbLang2 embeddingをそのまま再利用し、追加されたCDR-H3だけを同じモデル・入力整形・平均方法で推論する。配列とembeddingの対応を保存する。
4. 新しい3時点の全cloneをまとめて、元projectionと同じUMAP設定で再計算する。モデルや配列の追加学習はしない。
5. 現行と同じ帯域幅の決め方でKDEを計算する。点の集合とUMAPが変わるため、帯域幅の数値自体は元図と同じとは限らない。
6. 新しいUMAP座標から旧条件のcloneだけを取り出した比較も作る。この場合は両集合で同一の座標・KDE帯域幅・格子を使用する。

K-meansやTop Nの再選択は実行しない。背景図への入力条件を比較することが目的で、候補配列の順位への影響はこの処理では確認していない。

## 2. 実行方法

準備済みの専用Python環境で実行する。以下は説明用のパスであり、実際の保存先へ置き換える。

```powershell
.\.venv\Scripts\python.exe scripts\compute_d_filter_comparison.py `
  --run-dir "outputs\run_EXAMPLE" `
  --projection-dir "outputs\run_EXAMPLE\projection_EXAMPLE" `
  --model-dir "models\ABLANG-2-paired" `
  --output-dir "outputs\d_filter_comparison_EXAMPLE"

.\.venv\Scripts\python.exe scripts\render_d_filter_comparison.py `
  --comparison-dir "outputs\d_filter_comparison_EXAMPLE" `
  --reference-image "local_records\reference\provided_fig1c.png"
```

UMAPのキャッシュ保存先を指定する必要がある場合は、実行前に `$env:NUMBA_CACHE_DIR` を書込可能なローカルフォルダーへ設定する。参照画像は利用者が用意したFig.1cを使う。rendererの `--font-path` で日本語フォントを指定でき、既定はWindowsのMeiryo。

D検査を使った旧policy `takara-rg-hIGH20181210-v1` の保存済みRG Excel run専用の補助解析であり、CPM CSV用のD判定を新しく作る機能ではない。現在の入力処理で読み込んだ元情報から旧D条件を再適用し、旧clone集合と順序が再現されることを検証する。すでにDを採否に使わない新policyのrunは比較元として受け付けない。元モデル・package・poolingとcacheを照合する。UMAP/KDEの実行環境も元projectionと比較し、版が異なる場合は環境差を解消してから条件の影響を評価する。

出力先は存在しない新しいフォルダーにする。途中で停止すると部分的な出力が残ることがあるため、`comparison_inputs.json` の `status: completed` と、描画後の `render_metadata.json` を確認する。既存結果の上書きや中断箇所からの再開は行わない。

## 3. 2種類の図の意味

| 図 | 表示と比較できること | 限界 |
| --- | --- | --- |
| `d_filter_independent_umap_comparison.png` | 論文参照、保存済み旧条件の背景、Dを採否に使わず再計算した背景の3段 | 段ごとに独立UMAPと密度尺度。段をまたぐ位置・色の数値は直接比較できない |
| `d_filter_fixed_coordinate_comparison.png` | 新しい全集合のUMAP上で、旧条件の集合と新しい全集合を上下に表示。6面で同一の軸・格子・帯域幅・色尺度 | 上段も新座標による再表示であり、以前のUMAP図そのものではない |

各面はその集合・時点のcloneを等重みとし、格子上の密度積分が1となるよう正規化する。明るい色は相対的な集中を表し、点数が増えたこと自体を全面的な明るさにはしない。

新旧の背景には候補点を描かない。上段の提供論文画像に含まれる赤点はCoV-AbDab関連配列なので、新旧の背景と同じ強調対象ではない。論文に似た外形になったかと、原解析の入力・条件・候補が一致したかを区別する。

## 4. 出力と保護

配列index、embedding、cloneと元行の対応、旧集合から新集合への対応index、座標・密度、設定・hash、図を専用出力先へ保存する。原本と旧runは読み取りのみで、前後のhashを照合する。補助スクリプトの追加で標準adapterやGUIの採用条件を変えない。

全出力は研究データを含み得るためGit管理外にする。GitHubに記録するのは汎用スクリプト、手順、判断理由、検証範囲だけである。個別の実行件数・図・検体情報・hash・実データ監査は公開しない。
