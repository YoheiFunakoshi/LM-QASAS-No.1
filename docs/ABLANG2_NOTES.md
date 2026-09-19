# AbLang2の出所・設定・未確定事項

確認日：2026-09-19。AbLang2の公式コードとPyPI配布物に基づく記録です。

## 入手先

- package：[PyPI ablang2 0.2.1](https://pypi.org/project/ablang2/0.2.1/)
- 公式コード：[oxpig/AbLang2](https://github.com/oxpig/AbLang2)
- 調査時のGitHub commit：586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3
- checkpoint：[Zenodo record 10185169](https://zenodo.org/records/10185169)、ablang2-weights.tar.gz
- 抗体モデル名：ablang2-paired。ablang1-heavyとtcrlang-pairedは今回のモデルではない。

APIキー・アカウント認証の不要な公開配布物を使います。モデル取得時には外部へ接続しますが、確認した推論経路はローカルPyTorch演算です。レパトアを送信するAPIは使いません。この説明は、Codex/ChatGPT自体のデータ保持・学習設定を変更するものではありません。

## バージョンとライセンス

PyPIの0.2.1を採用します。調査時のGitHub setup.py表記は0.1.1のため、GitHubの表記から導入済み版を推測しません。導入版と依存は実測して固定します。

コードのLICENSEはBSD-3-Clause、モデルのZenodo metadataのlicense IDはbsd-3-clause-clearです。区別して記録し、現時点ではモデルや第三者コードを本リポジトリで再配布しません。本プロジェクト独自コードの公開ライセンスはまだ選定していません。

PyPI wheelの公開SHA-256：
`eda9b550bd8d9bdf1caaac821c3951deff5ef7141d4644292c6ba09126bbd834`

モデルarchiveの公開サイズ：166154117 bytes。MD5：
`6425b35e9b83750fde67a6a6d240d996`

モデル設定の hidden_embed_size は480です。取得時の完全なarchiveと展開後ファイルのSHA-256は、ローカルmanifestと人工配列試験の記録に残します。

## 今回のembedding定義

- AbLang2 0.2.1、学習済みablang2-paired、CPU。モデルparameterはtorch.float32、返却embeddingはNumPy float64と実測し、別々に記録する。
- 入力：各人工CDR-H3を重鎖側、軽鎖側を空文字列としたペア。
- mode=seqcoding、align=False、fragmented=False。
- 公式wrapperが入力を `<CDR3>|` に整形する。
- 公式seqcodingはCDR3残基に加えて `<`、`>`、`|` の3個の特殊トークンを含む平均。paddingは除外する。
- この平均方法を、残基だけの平均と同一視しない。

元のLM-QASASで同じcheckpoint・poolingを使ったかは未確認です。480次元が得られたことだけで論文の厳密な再現成功とはしません。人工CDR-H3様文字列での動作検証であり、生物学的性能の検証ではありません。

## 0.2.1実装上の注意

- ローカルモデルパスは文字列内に大文字の ABLANG- を含める必要がある。専用の一般的なlocal_files_only引数はない。
- 通常のモデル名指定では、package内に重みがなければ自動取得する。本プロジェクトでは取得と推論を分け、検証済みローカルパスから読む。
- fragmented引数はformat_seq_input内でadd_extra_tokensに渡されない。そのため今回のalign=False経路では、fragmented=Trueを指定しただけで特殊トークンを外せるとは判断しない。
- pretrainedのrandom_init引数もloaderへ渡されない。今回の学習済み重みの使用には影響しない。
- ncpuはPyTorchのthread数とは別なので、torch.set_num_threadsで明示する。
- align=True用のANARCI/Pandasは今回不要。

参照：[loader](https://github.com/oxpig/AbLang2/blob/586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3/ablang2/load_model.py)、[入力wrapper](https://github.com/oxpig/AbLang2/blob/586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3/ablang2/pretrained.py)、[encoding](https://github.com/oxpig/AbLang2/blob/586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3/ablang2/pretrained_utils/encodings.py)、[公式notebook](https://github.com/oxpig/AbLang2/blob/586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3/notebooks/pretrained_module.ipynb)。

## GPU対応

今回の検証はCPUで完結しました。GTX 1660 SUPER / driver 457.51は検出済みですが、現在のCPU版PyTorchではGPU可否を判断できません。

CUDA 12/13系の新しい構成に移行するには、ドライバーとの互換性確認が必要です。古いCUDA 11系には別の互換条件があるため、単に「GPUは使えない」とは結論しません。ドライバー更新やCUDA環境変更は今回行っていません。

参照：[NVIDIA互換性資料](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[PyTorch公式導入手順](https://pytorch.org/get-started/locally/)。

## 修士論文を追加確認した結果

2025年の修士論文の本文pp27–28（PDF pp38–39）に、公開事前学習済みAbLang2-paired、480次元、各残基embeddingの平均という記載がありました。モデル選択の補助根拠になりますが、特殊トークンや正確なcheckpoint/層の扱いを決定できる元コードは得られていません。

最新論文・ポスターを優先し、公式seqcodingの特殊トークン込み平均と同一だったかは引き続き要確認です。今後の本解析では採用するpoolingを明示し、必要なら残基のみの平均との差を比較します。
