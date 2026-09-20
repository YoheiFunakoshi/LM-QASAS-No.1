# LM-QASAS No.1

1人分のPre・Peak・PostのBCRレパトアから、AbLang2と時系列の変化を用いて抗原応答性の候補配列を抽出する、Windows向けローカルアプリの開発プロジェクトです。

## 現在の状態

**Pre・Peak・Postの3ファイル投入、AbLang2・K-means、共通UMAP画像、候補数の変更、CSV/PNG保存を行うローカル操作画面を実装しました（2026-09-20）。入力はCPM CSV、または対応するタカラ／RGレポートExcel（Back_data）です。3時点は同じ形式でそろえます。検証の範囲は[開発記録](docs/DEVELOPMENT_LOG.md)に記載します。k-NN、GPU推論、複数人のDB統合は今後の段階です。**
後から振り返るための入口は **[日本語解説書](docs/GUIDE_JA.md)** です。仕組み、計算例、設定の理由、操作、出力の見方、GitHubで履歴をたどる方法をまとめています。

## 最初に作るもの

環境準備済みのPCでは、プロジェクト直下の **`Start-LMQASAS.cmd` をダブルクリック**します。このPC内の `http://127.0.0.1:8765` が開きます。初めて環境を作る場合は[準備手順](docs/SETUP.md)、操作の説明は[日本語解説書の6節](docs/GUIDE_JA.md#6-実行と出力数だけを変える操作)を参照してください。処理完了後、起動した端末でCtrl+Cを押して終了します。

1. 1人分のPre・Peak・Postの3ファイルを読み込む（CPM `.csv` 3つ、または対応レポート `.xlsx` 3つ）。
2. CDR-H3アミノ酸配列をAbLang2で数値化し、処理状況と比較画像を表示する。
3. 3時点を比較して候補を順位付けする。
4. 上位300・500・1,000種類、または任意の正整数で指定した種類数を一覧表示し、保存する。

Top Nは、重複を除いたCDR-H3アミノ酸配列の種類数です。同じCDR-H3に対応する元clone情報は保持します。解析用cloneをCDR-H3だけで事前統合することはしません。

## 将来の利用

複数人の候補を集め、QASASで照合する擬似参照データベースを作ります。例として5人から各1,000件を集めると延べ5,000件となりますが、同一配列の統合後の種類数は異なり得ます。この統合機能とQASAS用出力形式は後の開発段階で扱います。

## 準備と開発計画

- [日本語解説書：考え方から実行・結果の読み方まで](docs/GUIDE_JA.md)
- [入力形式・CPM CSVの規則と未検証のfilter](docs/INPUT_RULES.md)
- [タカラ／RGレポートExcelの対応形式・採用条件](docs/TAKARA_INPUT.md)
- [開発計画と次の判断](docs/PLAN.md)
- [Windowsの準備・再実行手順](docs/SETUP.md)
- [AbLang2の出所・pooling・未確定事項](docs/ABLANG2_NOTES.md)
- [資料の優先順位と旧手法との差分](docs/SOURCE_COMPARISON.md)
- [CDR-H3候補の件数指定](docs/SELECTION.md)
- [共通UMAP・密度・候補の印の読み方](docs/VISUALIZATION.md)
- [論文 Fig. 1cとの比較・再現性の確認方法](docs/FIGURE_COMPARISON.md)

公式モデルをローカルに取得済みで、APIキーは不要です。解析はCPU上で実行します。同じスコアから候補数だけを変更する再選択機能もあります。pooling等の厳密な論文条件との一致、および候補の抗原特異性・抽出性能は未検証です。

Excel対応はRepertoire GenesisのHuman IGHレポート `hIGH20181210` の特定レイアウトに限ります。2枚目の `Back_data` の全データを読み、上位50件のランキングは入力に使いません。配列の末端や存在しないNT情報を補わず、CPMとは別の採用条件と監査を記録します。任意のタカラ製品・RGレポートに対応するものではありません。

Excelの新規解析は、利用者の明示承認によりDの注釈・機能を採否に使わないpolicy `takara-rg-hIGH20181210-v2-ignore-d` を使用します。V/J・読み枠・CDR3・C/isotypeの条件は維持し、Dの元情報とレポート集計照合は残します。旧runの条件は変わりません。新条件で候補を得るには3ファイルから新しい解析を開始します。[条件と理由](docs/TAKARA_INPUT.md)

## 記録の入口

- [仕様と科学的な前提](docs/SPECIFICATION.md)
- [決定事項と未決定事項](docs/DECISIONS.md)
- [開発・検証の記録](docs/DEVELOPMENT_LOG.md)
- [作業手順と再開方法](docs/WORKFLOW.md)

## 研究データの扱い

原本を削除・改変・上書きしません。前処理はメモリ内または別の解析用コピーに適用し、結果とログは専用出力先に保存します。

この公開リポジトリではコードと手順を管理します。実配列、被験者情報、実データ由来の候補一覧・画像・embedding、個人別集計、実データのハッシュ、提供された資料ファイルは登録しません。詳細な実データ監査記録はGit管理外のローカル領域に保存します。

Google Driveにはアクセスしない、という現在のユーザー指示を維持します。提供されたローカルデータの解析利用は許可されていますが、GitHub公開の許可とは区別します。

## 参考論文

Masuda et al. *LM-QASAS: reference-free identification of antigen-specific sequences from the BCR repertoire using antibody language models.* Frontiers in Immunology (2026). DOI: [10.3389/fimmu.2026.1844788](https://doi.org/10.3389/fimmu.2026.1844788)

候補抽出結果は抗原応答性の検証対象であり、個々の配列の抗原結合能・中和能を確定するものではありません。
