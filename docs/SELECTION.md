# 重複しないCDR-H3候補の件数指定

2026-09-19、ユーザーが「同じCDR-H3アミノ酸配列を重複させずN種類集める」と確定しました。選択肢は300、500、1000、さらに任意の正整数です。100も任意指定できます。

## 実装した範囲

src/lmqasas/selection.py は、**すでにスコアが付いた1人分のPeak clone候補**を受け取り、異なるCDR-H3を上位N種類返す独立した部品です。AbLang2、clone生成、LM-QASASスコア計算、CSV出力、GUIはこの部品には含まれません。

解析ではV/J/CDR-H3/isotypeで区別するcloneを保持し、最終出力でCDR-H3を一意化します。同じ配列のembeddingをキャッシュしても、解析用のclone観測点を消さない設計です。

## インターフェース

```python
from lmqasas.selection import TOP_N_PRESETS, select_unique_cdr3

# Synthetic examples only. These are not validated biological candidates.
records = [
    {"cdr3": "CAAAF", "score": 8.0, "source_row": 1, "timepoint": "Peak"},
    {"cdr3": "CAAAF", "score": 8.0, "source_row": 2, "timepoint": "Peak"},
    {"cdr3": "CASSF", "score": 5.0, "source_row": 3, "timepoint": "Peak"},
]
result = select_unique_cdr3(records, requested=300)
# result.available == 2, result.returned == 2, result.shortfall == 298
```

必要な列はcdr3、score、source_row、timepointです。timepointは厳密にPeakである必要があります。元ファイル名、V/J、isotype等の追加情報は、各配列のclone_recordsへまとめて保持します。利用側が同一被験者・同じ採点方法のレコードを渡し、source_rowの意味と一意性を管理します。

配列は大文字の標準20アミノ酸のみを受け付けます。空白除去・大文字化・配列修復は行いません。CDR-H3の長さ等の生物学的filterは上流の入力検証で扱います。

## 暫定の実装判断

次はユーザーが委任した範囲で採用した、論文由来とは断定しない規則です。

- 同じCDR-H3に複数scoreがある場合は最大値を代表scoreとする。低いscoreの元cloneも消さず、全由来を保持する。
- score降順、同点ではCDR-H3文字列の辞書順にする。同点内の順位は生物学的な優劣を意味しない。
- 同点がN件の境界を跨ぐ場合もN種類に収め、境界score、選択数、除外数を返す。
- scoreは有限な実数としてfloatへ変換し、同点は丸めを加えない完全一致で判定する。
- 候補不足時は得られる種類だけを返し、requested/available/returned/shortfallを明示する。複製で水増ししない。

入力レコードは変更せず、出力の由来レコードはdeep copyで分離します。Pre/Post混入、NaN/Inf、boolや小数のNなどを拒否します。例外に配列内容を埋め込みません。

## 検証

人工配列のみの13テストで、重複除外と全由来保持、順位、境界同点、不足、不正値、入力不変、300/500/1000/任意17件を確認しました。既存4件を含む17テストがすべて成功しました。

この検証は候補選択部品の正しさの確認であり、抗原特異性や実データの候補抽出性能の検証ではありません。
