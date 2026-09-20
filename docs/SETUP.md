# Windowsでの準備と再実行

このページは環境構築・確認と人工配列の最小試験の手順です。3時点を解析するGUI・コマンドの操作は[日本語解説書](GUIDE_JA.md)の6節にあります。環境準備後はプロジェクト直下の `Start-LMQASAS.cmd` から起動できます。

## 検証済みの構成

2026-09-19、Windows 11 x64、Python 3.12.1、PyTorch 2.14.0+cpu、AbLang2 0.2.1、NumPy 2.5.3で確認しました。モデルparameterはtorch.float32、公式seqcodingの返却embeddingはNumPy float64と実測しました。CPU・4 threadsです。依存バージョンは [requirements-cpu.txt](../requirements-cpu.txt) に固定しています。

GTX 1660 SUPER 6GiBとドライバー457.51を検出しましたが、今回のPyTorchはCPU版です。CUDA available=Falseは今回の構成として想定どおりで、GPU故障を意味しません。GPU版の導入・互換性検証・速度測定は未実施です。

## 1. 専用環境を作る

PowerShellをプロジェクトのルートで開きます。既存の .venv がある場合は作り直さず、下の「既存環境を更新する」を行ってから手順2へ進めます。既存の別プロジェクトのPython環境にはインストールしません。

```powershell
py -3.12 -m venv .venv
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe -m pip --isolated install pip==26.2.1
& .\.venv\Scripts\python.exe -m pip --isolated install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu -c requirements-cpu.txt
& .\.venv\Scripts\python.exe -m pip --isolated install -r requirements-cpu.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pip --isolated install --no-deps --no-build-isolation -e .
```

PyTorchを先に公式CPU indexから入れてから、残りの固定依存をPyPIから導入します。pip自体は実行時依存とは別に版を記録しています。上記は検証済み環境の再作成用であり、他OSや他Python版の互換性を保証するものではありません。

### 既存環境を更新する

K-means用scikit-learn、共通UMAP用umap-learn、PNG描画用matplotlibに加え、対応するタカラ／RGレポートExcelの読込に必要なpackageを追加しました。以前の環境にも次を実行します。モデルの取得より前にpackageをinstallしてください。

```powershell
& .\.venv\Scripts\python.exe -m pip --isolated install -r requirements-cpu.txt
& .\.venv\Scripts\python.exe -m pip --isolated install --no-deps --no-build-isolation -e .
& .\.venv\Scripts\python.exe -m pip check
```

追加版：scikit-learn 1.9.1、SciPy 1.18.1、threadpoolctl 3.7.0、joblib 1.6.0、cloudpickle 3.1.2、narwhals 2.26.0。既存のPyTorch/AbLang2/NumPyの版は維持しています。

可視化追加版：umap-learn 0.5.12、matplotlib 3.11.2、numba 0.67.0、pynndescent 0.6.0。間接依存もrequirementsに固定しました。GUIサーバーはPython標準ライブラリ、画面はローカルHTML/CSS/JavaScriptを使用し、Node.jsや外部Webサービスの契約は起動に不要です。Numbaの初回コンパイルは時間がかかるため、初回だけで毎回の速度を判断しません。

Excel読込追加版：openpyxl 3.1.5、et_xmlfile 2.0.0、defusedxml 0.7.1。Excel本体のインストールやCOM操作は使いません。openpyxlの読み取り専用モードでブックを開き、XML読込にはdefusedxmlを必須とします。入力ブックの保存・数式再計算・外部リンク更新は行いません。対応する版・レイアウトは[Excel入力の規則](TAKARA_INPUT.md)、動作検証の範囲は[開発記録](DEVELOPMENT_LOG.md)を参照してください。

## 2. 環境を確認する

```powershell
& .\.venv\Scripts\python.exe scripts/environment_check.py --require-packages
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

環境のJSONは local_records/ に保存します。CPU版でも必要なpackageがimportできれば成功します。既存の記録は上書きしません。

## 3. モデルを一度取得する

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_ablang2.py
```

公式Zenodoの公開モデル約166MBを取得し、公開されたサイズ・MD5を照合します。展開後の2ファイルのSHA-256、出所、モデル設定を manifest.json に保存します。保存先は models/ABLANG-2-paired/ です。

再実行時は既存モデルのhashを確認し、正常なら再ダウンロードしません。不一致を検知した場合は停止します。ダウンロードしたarchiveは models/.ablang-download-*/ に保持します。取得失敗時も調査用ファイルを保持するため、繰り返し失敗する場合は状況を確認してから再試行します。

models/ はGit管理外です。個人のレパトア配列を入力する処理は、この取得スクリプトにありません。

## 4. 人工配列10本を試す

```powershell
& .\.venv\Scripts\python.exe scripts/smoke_ablang2.py --device cpu --batch-size 4 --threads 4
```

このコマンドはコード内で人工配列を生成し、実レパトアを読み込みません。ローカルのモデルを指定し、import完了後のPythonソケット通信を禁止して推論します。OS全体のネットワーク遮断を行う機能ではありません。

local_records/smoke_<日時>/ に数値配列と実行条件JSONを保存します。確認項目は、10×480の形状、NaN/Infなし、同条件での再現、batch 4対1の一致、保存・読み戻しの一致です。各runは別フォルダーです。

## 初期の人工配列試験の結果と限界

全項目が成功しました。初回のモデル読込は約0.40秒、10本×3回の推論は計約0.77秒でした。全件レパトアの実行時間や候補抽出の正確性を示す測定ではありません。

poolingの定義と論文条件の未確定点は [AbLang2確認事項](ABLANG2_NOTES.md) を参照してください。最終版スクリプトの人工データ検証記録は [CPU検証記録](validation/phase01_cpu.json) に保存します。実データの解析結果はこの場所へ保存しません。
