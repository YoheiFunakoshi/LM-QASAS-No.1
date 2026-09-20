# 入力検査・clone生成の規則

対象は`src/lmqasas/inputs.py`の`input-v2`です。1人分のPre・Peak・Postを読み取り、採用した行から各時点のcloneを作ります。元ファイルは変更しません。以下は**現在の実装条件**であり、元論文の前処理を完全に再現できたことを意味しません。

## 対応形式と共通条件

異なる3つの実体ファイルを、利用者がPre・Peak・Postとして指定します。同じファイルの重複指定や、CSVとExcelの混在は受け付けません。拡張子で読込処理を選び、内容が対応する形式かを検査します。ファイル名や報告日から時点を推定しません。

| 入力形式 | 拡張子 | 形式別policy | 詳細 |
| --- | --- | --- | --- |
| CPM CSV | `.csv` | `cpm-csv-input-v1` | 本ページの1〜6節。従来の採用条件を維持 |
| タカラ／RGレポートExcel（Back_data） | `.xlsx` | `takara-rg-hIGH20181210-v2-ignore-d` | [対応するHuman IGHレポートの規則](TAKARA_INPUT.md) |

どちらも時点内のclone keyはV/J候補集合・CDR3・isotype/subclassで、unique cloneを等重みで扱います。ただし、元の列・注釈と採用条件は形式ごとに異なります。Excelの`in-frame`をCSVの`WithConserved_NoStop`へ置換しません。Excel用のNTlengthやFrequencyを作ってCSV条件を通すこともありません。形式とpolicyは入力監査に残します。

Excelの新policyではDの注釈・機能を採否とclone keyに使いません。V/Jなど他の条件と、Dを含む供給元の集計照合は維持します。旧policyの結果を新条件へ書き換えず、新規解析として実行します。本プロジェクトが対応するCPM CSV仕様にはD列がなく、CPMの入力条件は従来どおりです。詳細は[Excel規則](TAKARA_INPUT.md)を参照してください。

## 1. CPM CSVのファイルと列

- 異なる3つの実体ファイルを、利用者がPre・Peak・Postとして指定します。同じファイルを別時点へ重複指定すると停止します。
- 文字コードはUTF-8です。UTF-8 BOM付きも受け付けます。他の文字コードを推測して読み直しません。
- 次の9列がそれぞれ1つ必要です。列順は任意ですが、余分な列、欠損した列、重複列は受け付けません。

```text
Vseg, Jseg, CDR3, AAlength, NTlength, Type, Cseg, Counts, Frequency(%)
```

CSV構文の破損、行の列数不一致、ヘッダーの不一致はファイル単位のエラーとして停止します。下記の値検査に失敗した行は除外し、理由と元行を監査へ記録します。いずれかの時点で採用cloneが0になる場合も停止します。

## 2. CPM CSVで論文の要件を確認できる範囲

| 項目・論文の要件 | 今回の実装 | 判断の根拠・未確認事項 |
| --- | --- | --- |
| CDR-H3アミノ酸配列 | 大文字の標準20アミノ酸のみ。空文字、空白、小文字、`*`や`X`などは除外 | 勝手な大文字化・空白除去・配列修復はしない |
| CDR-H3長5未満の除外 | 実配列長が5未満なら除外 | 最新論文の長さ条件を採用 |
| 記載長との整合 | `AAlength`は数字だけの整数表記で、実際のCDR3長と一致することを確認 | 入力表の整合性検査。配列を記載長へ修正しない |
| in-frame | `NTlength`が数字だけの整数表記で、実際のCDR3長の3倍であることを確認 | **暫定のCDR3長整合性検査**。塩基配列がないため、翻訳の一致や全VDJの読み枠を独立に検証できない |
| stopなし・品質ラベル | `Type`が`WithConserved_NoStop`に完全一致し、CDR3文字も適切な行を採用候補とする | 暫定の入力条件。Typeは入力元のラベルであり、全VDJにstopがないことの独立した証明ではない |
| functional V/D/J | V/Jの表記形式を確認する | D gene・機能分類の列がなく、functional V/D/Jを検証できない。遺伝子名の形式が合うだけで機能性を認定しない |
| ORF/pseudogene除外 | 独立した機能注釈との照合は未実装 | 該当行を確実に除外したとは表示しない |
| clone = V/J/CDR-H3/isotype | 下記のgene候補集合・CDR3・Cseg由来のsubclassで時点内を集約 | allele除去、候補集合、subclass粒度は今回の暫定条件。元解析と完全一致するかは要確認 |
| unique cloneを等重みで扱う | 各時点のunique cloneを1点として使用 | Counts・Frequencyによる重み付け、Countsによる点の複製はしない |
| Counts・Frequencyの保存 | Countsは有限で0以上、Frequencyは有限で0以上100以下を確認 | Countsを整数に限定せず、単位をread/UMIと推測しない。Counts閾値のfilterや頻度の再計算はしない |

`Type`はisotypeとして利用しません。`WithConserved_NoStop`以外の値を、名前から似た意味だと推測して受け入れることもしません。この採用条件は原資料の完全再現条件とは区別し、監査に`accepted_types`として保存します。

監査には`full_vdj_functionality_verified: false`を明記します。入力表から確認できることと、上流の注釈処理で済んでいる可能性があることを区別するためです。

## 3. CPM CSVのV/J候補の正規化と曖昧な注釈

Vsegは`IGHV`、Jsegは`IGHJ`で始まる所定の形式を検査します。実在する全遺伝子の辞書やfunctional状態との照合ではありません。

clone keyを作る際には、各注釈の前後の空白とallele表記を外します。複数候補は`//`で分割し、gene名の重複を除いて並べ替えます。

```text
説明用の注釈例:
IGHV1-3*02 // IGHV1-2*01  →  IGHV1-2//IGHV1-3
```

これは複数候補から1つを選ぶ処理ではありません。未解決の候補集合をそのまま1つの注釈keyとして保持します。

- 候補の並び順やalleleだけが異なり、gene候補集合が同じなら同じkeyになります。
- `IGHV1-2`と`IGHV1-2//IGHV1-3`は異なるkeyです。候補が一部重なるだけでは同一としません。
- 複数候補の区切りは`//`のみとして扱い、所定の形式に合わない注釈は除外します。単独の`/`は遺伝子名の一部として残り得るため、候補区切りだと推測して分割しません。
- 元のVseg/Jseg文字列は、採用行の`raw_records`に保存します。

論文のgene単位という定義に合わせつつ、不確かな注釈を確定したように扱わないための暫定方針です。元解析の注釈正規化と完全に一致するかは要確認です。

## 4. CPM CSVのCsegとisotype/subclass

Csegから次の対応だけを明示的に認めます。表記前後の空白を除き、形式が適切なallele表記を外します。

| 受け付けるCsegの基本表記 | 保存するisotype値 |
| --- | --- |
| `IGHG1`〜`IGHG4`、`IgG1`〜`IgG4` | `IGHG1`〜`IGHG4` |
| `IGHA1`・`IGHA2`、`IgA1`・`IgA2` | `IGHA1`・`IGHA2` |
| `IGHM`・`IgM` | `IGHM` |
| `IGHD`・`IgD` | `IGHD` |
| `IGHE`・`IgE` | `IGHE` |

IgG/IgAをsubclassなしの値へまとめません。たとえばIGHG1とIGHG2は別のclone keyです。subclassを判別できない`IgG`・`IgA`、未対応値、複数候補を含むCsegは除外し、`cseg_unmapped`を記録します。

**ファイル名はisotypeを決める情報として使いません。** たとえばファイル名に「IgG」が含まれていても、Csegが`IGHA1`なら、現在の実装では`IGHA1`として扱います。ファイル名だけを根拠にIgGへ書き換えたり、IgAの行を自動除外したりしません。これは不一致の原因が解明されたという意味ではなく、入力行の明示的な注釈を保つ暫定判断です。検体調製・ファイル名の意味とCsegの関係は要確認として残します。

## 5. CPM CSVのclone生成と元データへの対応

採用行を、時点ごとに次の4項目で集約します。

```text
(V gene候補集合, J gene候補集合, CDR3アミノ酸配列, Cseg由来のisotype/subclass)
```

同じkeyの入力行が複数あっても、その時点の解析では1点です。Countsを足して点の数や重みを増やすことはありません。同じkeyがPreとPeakにあれば、各時点に1点ずつ残します。

各cloneにsubject、timepoint、source_file、source_row、source_rows、raw_recordsを保存します。`source_row`は最初に採用された行、`source_rows`は集約された全行です。行番号はヘッダーを1行目とする物理的な開始行番号で、引用符内に改行のあるCSVでも元の開始位置を指します。

採用した元行の文字列はJSONに保存します。除外行は元行番号と理由を記録し、その内容は保護された元CSVから参照します。任意の元入力文字列をそのまま候補CSVの列へ展開しません。

## 6. CPM CSVの入力監査とエラー時の対応

`input_audit.json`には、時点別の総入力行数、採用・除外行数、unique clone数、集約された行数、複数V/J候補のある行数、除外理由別の数を保存します。1行に複数の問題があれば全理由を記録するため、理由別件数の合計は除外行数より大きくなり得ます。

主な除外理由は、`cdr3_noncanonical`、`cdr3_too_short`、`aa_length_invalid`／`aa_length_mismatch`、`nt_length_invalid`／`nt_length_mismatch`、`type_not_accepted`、`v_annotation_invalid`、`j_annotation_invalid`、`cseg_unmapped`、`counts_invalid`、`frequency_invalid`です。

読み取り前後のSHA-256を比較し、処理中に入力が変更された場合は停止します。解析全体でも最後に入力hashを照合します。CSVが読めたことだけで、原本の状態確認を完了とはしません。

入力検査でrunフォルダー作成前に停止した場合、`input_audit.json`が保存されないことがあります。エラーが出たからといって原本を直接修正せず、入力仕様の追加または別の解析用コピーへの変換が必要かを確認し、その理由と手順を記録します。現在の採用条件を変更した場合はpolicy versionと決定記録・テストも更新します。

実際のファイル名・実配列・被験者情報・件数・入力hashを含む監査は、Git管理外のローカル領域に保存します。本ページは規則と説明用の例だけを公開します。
