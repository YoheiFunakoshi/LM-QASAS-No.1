# LM-QASAS No.1

1人分のPre・Peak・PostのBCRレパトアから、AbLang2と時系列の変化を用いて抗原応答性の候補配列を抽出する、Windows向けローカルアプリの開発プロジェクトです。

## 現在の状態

**入力CSVからAbLang2・K-means・候補CSV保存までのコマンド実行エンジンを実装しました（2026-09-19）。提供された3時点の実データで、候補CSV保存までの完走を確認済みです。共通UMAP画像・GUI・k-NNは未実装です。**
後から振り返るための入口は **[日本語解説書](docs/GUIDE_JA.md)** です。仕組み、計算例、設定の理由、操作、出力の見方、GitHubで履歴をたどる方法をまとめています。

## 最初に作るもの

1. 1人分のPre・Peak・Postの3ファイルを読み込む。
2. CDR-H3アミノ酸配列をAbLang2で数値化し、処理状況と比較画像を表示する。
3. 3時点を比較して候補を順位付けする。
4. 上位300・500・1,000種類、または任意の正整数で指定した種類数を一覧表示し、保存する。

Top Nは、重複を除いたCDR-H3アミノ酸配列の種類数です。同じCDR-H3に対応する元clone情報は保持します。解析用cloneをCDR-H3だけで事前統合することはしません。

## 将来の利用

複数人の候補を集め、QASASで照合する擬似参照データベースを作ります。例として5人から各1,000件を集めると延べ5,000件となりますが、同一配列の統合後の種類数は異なり得ます。この統合機能とQASAS用出力形式は後の開発段階で扱います。

## 準備と開発計画

- [日本語解説書：考え方から実行・結果の読み方まで](docs/GUIDE_JA.md)
- [CSV入力の規則と未検証のfilter](docs/INPUT_RULES.md)
- [開発計画と次の判断](docs/PLAN.md)
- [Windowsの準備・再実行手順](docs/SETUP.md)
- [AbLang2の出所・pooling・未確定事項](docs/ABLANG2_NOTES.md)
- [資料の優先順位と旧手法との差分](docs/SOURCE_COMPARISON.md)
- [CDR-H3候補の件数指定](docs/SELECTION.md)

公式モデルをローカルに取得済みで、APIキーは不要です。解析はCPU上で実行します。同じスコアから候補数だけを変更する再選択機能もあります。pooling等の厳密な論文条件との一致、および候補の抗原特異性・抽出性能は未検証です。

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
