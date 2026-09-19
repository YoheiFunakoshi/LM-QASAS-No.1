# Windowsでの準備と再実行

このページは環境確認と**人工配列だけの最小試験**の手順です。候補抽出アプリはまだ完成していません。

## 検証済みの構成

2026-09-19、Windows 11 x64、Python 3.12.1、PyTorch 2.14.0+cpu、AbLang2 0.2.1、NumPy 2.5.3で確認しました。モデルparameterはtorch.float32、公式seqcodingの返却embeddingはNumPy float64と実測しました。CPU・4 threadsです。依存バージョンは [requirements-cpu.txt](../requirements-cpu.txt) に固定しています。

GTX 1660 SUPER 6GiBとドライバー457.51を検出しましたが、今回のPyTorchはCPU版です。CUDA available=Falseは今回の構成として想定どおりで、GPU故障を意味しません。GPU版の導入・互換性検証・速度測定は未実施です。

## 1. 専用環境を作る

PowerShellをプロジェクトのルートで開きます。既存の .venv がある場合は作り直さず、手順2の確認から進めます。既存の別プロジェクトのPython環境にはインストールしません。

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

既存の準備環境にも、追加した候補選択部品を使う前に最後のeditable installを一度実行します。追加の外部依存はありません。

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

## 今回の結果と限界

全項目が成功しました。初回のモデル読込は約0.40秒、10本×3回の推論は計約0.77秒でした。全件レパトアの実行時間や候補抽出の正確性を示す測定ではありません。

poolingの定義と論文条件の未確定点は [AbLang2確認事項](ABLANG2_NOTES.md) を参照してください。最終版スクリプトの人工データ検証記録は [CPU検証記録](validation/phase01_cpu.json) に保存します。実データの解析結果はこの場所へ保存しません。
