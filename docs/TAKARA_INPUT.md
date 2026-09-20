# タカラ／RGレポートExcelの入力規則

対応するHuman IGHレポートを、CPM CSVへ変換せず直接読み込みます。画面での案内名は「タカラ／RGレポートExcel（Back_data）」です。実際に認識するのは、Repertoire Genesisの `Repertoire Analysis Report (Human IGH)`、`Sheet ver. hIGH20181210` の下記レイアウトです。タカラ製品やRGレポート全般への対応を意味しません。

実装は `src/lmqasas/takara.py`、形式名は `takara_rg_xlsx`、形式別policyは `takara-rg-hIGH20181210-v1` です。共通の入力監査versionは `input-v2` です。既存CPM CSVの条件は[入力規則](INPUT_RULES.md)で別に説明します。実施済みの検証と未実施の範囲は[開発記録](DEVELOPMENT_LOG.md)を参照してください。

## 1. ファイル選択と対象シート

1人分のPre・Peak・Postに、異なる3つの `.xlsx` ファイルを指定します。CSVとExcelの混在、同じ実体ファイルの重複指定は受け付けません。画面では従来と同じ3つのファイル欄を使い、拡張子とブックの内容を自動で検査します。Pre・Peak・Postは利用者が指定し、ファイル名・報告日・シート内の識別情報から推定しません。

最初の2シートはこの順序で必要です。

| シート | 用途と確認内容 |
| --- | --- |
| 1枚目 `PRINT_hIGH` | `F1` のレポート名と `F5` 末尾の版表記を確認 |
| 2枚目 `Back_data` | `G:Q` の `All Data` を解析元とし、集計値・配置マーカーを確認 |

`Back_data!G:Q` は**1行目からデータが始まり、ヘッダー行はありません**。`T:AA` はin-frameの上位50件ランキングなので、全レパトアの入力には使いません。

保存されたシート寸法が途中で終わっていても、実際のデータがその先に存在する場合があります。読込前にopenpyxlの `reset_dimensions()` を呼び、保存された最大行を終端として切り捨てない設計です。元ブックの寸法や内容を書き換える処理ではありません。

## 2. 全データの列対応

| 列 | 入力元の情報 | 利用方法 |
| --- | --- | --- |
| G | V gene候補 | 注釈の形式検査とclone key |
| H | V機能ラベル | `F` への完全一致を要求 |
| I | D gene候補 | 注釈の形式検査。clone keyには含めない |
| J | D機能ラベル | `F` への完全一致を要求 |
| K | J gene候補 | 注釈の形式検査とclone key |
| L | J機能ラベル | `F` への完全一致を要求 |
| M | C gene・isotype/subclass候補 | 単一に決まる対応済みsubclassをclone keyへ使用 |
| N | C機能ラベル | 元情報として保存。`F` の採用条件は追加しない |
| O | 入力元のCDR3配列 | 標準20アミノ酸、長さ5以上を確認し、そのまま使用 |
| P | 入力元の読み枠ラベル | `in-frame` への完全一致を要求 |
| Q | Reads | 0以上の整数値として検査・保存。解析の重みには使わない |

G:Qに数式やExcelエラー値がある場合、またはQが適切な整数値でない場合はファイル単位で停止します。値を推測したり数式を再計算したりしません。配列・注釈・ラベルの採用条件を満たさない行は理由と元シート・元行を記録して除外し、いずれかの時点の採用cloneが0なら停止します。

## 3. 集計との照合

採用・除外のfilterとは別に、全データとレポートの集計を照合します。

| 全データから計算する値 | 比較する集計セル |
| --- | --- |
| Qの合計 | `Back_data!C5`（Assigned reads） |
| Pが厳密に `in-frame` の行のQ合計 | `Back_data!C6`（In frame） |
| in-frame行における元の `(V,D,J,CDR3,C)` の異なる組合せ数 | `Back_data!C7`（Unique reads (In frame)） |

C4:C7は保存された0以上の整数値が必要で、C4 ≥ C5 ≥ C6も確認します。全データと上記の集計が一致しなければ停止します。C7は元の注釈を使う集計で、in-frame行数やLM-QASASのclone数とは異なります。LM-QASASのclone keyにDを加えたり、注釈正規化後の件数でC7を置き換えたりしません。

ランキングの `%Reads` はin-frame read合計を分母とする表示ですが、G:Qに全行の頻度列はありません。現実装はランキングの頻度を全データへ転記せず、Frequencyも計算・補完しません。

## 4. 採用条件と注釈の正規化

採用行は、標準20アミノ酸のみで長さ5以上のCDR3、`frame=in-frame`、V/D/J機能ラベルがすべて `F`、形式の合うV/D/J候補、単一に決まる対応済みC subclassを満たす必要があります。CDR3の空白除去・大文字化・修復は行いません。

V/D/J候補の区切りはコンマです。候補の前後の空白とallele表記を外し、重複を除き、並べ替えた集合として扱います。遺伝子名の内部にある単独の `/` は分割しません。V/Jの複数候補から先頭だけを選ぶことはなく、集合全体をclone keyへ残します。内部のkeyでは集合を `//` で表しますが、このレポート入力に `//` 区切りがある場合は受け付けません。

D gene名末尾の小文字 `a` / `b` は、このレポートのコピー表記として形式検査で許容します。元注釈の大文字・小文字は変更せず、そのまま `raw_records` に保持します。Dは形式・機能ラベルの検査に使いますが、clone keyには含めません。

C候補は、[CPM CSVと同じ明示的なisotype対応表](INPUT_RULES.md#4-cpm-csvのcsegとisotypesubclass)を各候補に適用します。候補が複数あっても、alleleを外した後にすべて同じ対応済みsubclassへ決まる場合だけ採用します。複数のsubclass、未対応値、未解決値が含まれれば除外します。IgG/IgAのsubclassはまとめず、ファイル名からisotypeを強制しません。

採用行を時点内の `(V候補集合, J候補集合, CDR3, isotype/subclass)` で集約し、各unique cloneを1点とします。Qのread数や頻度による点の複製・重み付けは行いません。

## 5. 入力元の注釈で確認できる範囲

`F` と `in-frame` は、レポートで供給された注釈への採用条件です。これらを独立に再注釈・検証したという意味ではありません。特に、この形式にはCDR3塩基配列、全VDJ塩基配列、独立した全VDJ productive判定、全VDJのstop注釈がありません。`F` の意味を独立したベンダー資料で確認したことにもなりません。

CDR3はO列のまま使用します。末端C/Wの有無を新たな採用条件にせず、C/Wの追加・削除も行いません。このレポートとCPMのCDR3境界定義が同一かは未確認です。Excelの読み枠ラベルを `WithConserved_NoStop` に置換したり、配列長の3倍を入力元のNTlengthとして作ったりしません。

監査には以下の限界を明示します。

```text
full_vdj_functionality_verified: false
nt_length_verified: false
vendor_function_labels_independently_verified: false
cdr3_boundaries_modified: false
counts_used_as_weights: false
```

取り込みの整合性確認と、候補の抗原特異性・結合能の検証は別です。

## 6. 元データへの対応とエラー時の確認

各cloneに `source_file`、`source_sheet`、`source_columns`、最初の採用行を示す `source_row`、集約した全行の `source_rows`、採用行の元セル値を持つ `raw_records` を保存します。Excelの行番号は `Back_data` の実際の行番号で、1行目もデータです。入力元のV/D/J/C・機能ラベル・CDR3・frame・countを保存し、clone keyの正規化で元セル値を置き換えません。

`input_audit.json` は入力形式、policy、シート・列、寸法の再設定、採用・除外理由、レポート集計との照合結果を記録します。主な除外理由は `cdr3_noncanonical`、`cdr3_too_short`、`frame_not_in_frame`、`v_function_not_F`、`d_function_not_F`、`j_function_not_F`、`v_annotation_invalid`、`d_annotation_invalid`、`j_annotation_invalid`、`cseg_unmapped_or_ambiguous` です。1行に複数理由が付くことがあります。

読込前後と解析の終了時に入力hashを照合します。原本の保存・改変・数式計算・外部リンク更新は行いません。形式不一致、集計不一致、破損、上限超過などで停止した場合は、原本を直接直さず、対応形式と必要な情報を確認します。入力検査でrunフォルダー作成前に停止すると、監査ファイル自体が保存されないことがあります。

Excel読込には展開後合計256 MiB、ZIP内4,096項目、シート走査250,000行の上限があります。マクロを含むarchiveや重複したarchive内ファイル名は受け付けません。GUIにはさらに1ファイル20 MiB・送信全体64 MiBの上限があります。CLIでGUI上限を回避しても、Excel読込の上限と検査は維持されます。

実ファイル名、実配列、sample情報、個別の件数、hash、詳細監査は公開文書・GitHubへ含めず、Git管理外のローカル領域に保存します。

## 7. CLIでの指定例

追加の形式指定引数は不要です。次は架空のパスによる例で、モデルと環境は[準備手順](SETUP.md)に従って用意します。

```powershell
& .\.venv\Scripts\python.exe scripts/run_analysis.py `
  --pre 'C:\analysis-input\pre.xlsx' `
  --peak 'C:\analysis-input\peak.xlsx' `
  --post 'C:\analysis-input\post.xlsx' `
  --subject 'example-subject' `
  --top-n 1000 `
  --clusters 500
```

その他の解析条件・候補数変更・結果の確認は[日本語解説書](GUIDE_JA.md)の6〜7節と共通です。入力形式や採用条件を変える場合は、保存済みrunの再選択ではなく新しい解析を実行します。
