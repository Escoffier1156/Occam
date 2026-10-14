# Occam へのプロンプト：形式と注意点

2026-10-13。内容は `py/occam.py`・`py/knobs.py`・`py/render.py`・
`py/screenable.py` の実装から転記。**語彙を足したらこのファイルも直す。**
ずれたら実装が正しい。

この版が対応する変更：charge knob の引退（2026-10-11）、標的を位置で読む
（2026-10-12）、順位付けの標本化をやめた・順位付け要求に答える・届かなかった
最も近い仕様を出す（2026-10-13）。

本文の綴りは全行を parse に通して確かめてあります（ライブラリ11、nitrogen 6、
rotatable 10、weight 5、成果物3、strict 6、停止14、破棄8）。

```bash
occam "散文のプロンプト"          # 端末での利便性
occam --spec request.json         # 記録・提出用。こちらが正
```

---

## 1. 必須の4つ

これが無いと答えが出ない。

| | 書き方 | 無いと |
|---|---|---|
| **ライブラリ** | `cationic ZINC` など（下表） | `stopping: no library named` |
| **成果物** | `256 distinct scaffolds` | `stopping: no deliverable named` |
| **条件** | 下の knob から | **書かなければ**既定（最も厳しい設定）。**書いて語彙外なら停止** |
| **譲れるか** | `strictly` などを添える | **すべて「緩和してよい」になる** |

**「書かない」と「書いたが語彙外」は別です。**書かなければ最も厳しい設定が入り、
**それが `read as:` に `the request named no ... condition, so the strictest
registered setting was used` と明示されます。**黙って適用されることはありません。
語彙にない値（回転結合6、MW 500 など）を書いた場合は、丸めずに停止します。

### 成果物は「何個の何か」を書く

```
256 distinct scaffolds     ← 可
256 unique candidates      ← 可
256                        ← 不可。何の256個か分からない
```

**判定基準になるのは scaffold の数**で、生存数ではない。成果物が「1骨格1代表」なので、
制約は骨格の数に掛かる。

**骨格の定義：Bemis-Murcko 骨格**（RDKit `MurckoScaffoldSmiles`）。環系と、環どうしを
つなぐリンカーだけを残し、側鎖を落とした構造。**同一骨格は1件**と数えます。画面にも
`counted as Bemis-Murcko frameworks (RDKit MurckoScaffoldSmiles), one per distinct
framework` と出ます。判定がこの数で決まるので、定義が違えば答えが違います。

---

## 1.5 何を頼むか — 成果物を境目に置く

**この道具が返すのは frontier です。**「成立する／しない」の一点ではなく、*どの条件を
どれだけ動かせば成立するか*という境界。だから**成果物の数をどこに置くかで、返ってくる
情報量が変わります。**

同じライブラリ（-P、17,775件）、同じ条件（一級アミン strict、回転結合≤5、MW<450）で、
要求数だけを変えた実測です。

| 要求 | 返るもの | 読み取れること |
|---|---|---|
| 120骨格 | **要求どおり成立。**430分子・187骨格 | 成立は分かる。**どの条件が効いているかは分からない** |
| 256骨格 | 不成立。回転結合を7に緩めて成立（1,200分子・475骨格） | **1段で届く。**動かすべき軸が1つに特定される |
| 500骨格 | 不成立。回転結合を9に緩めて成立（1,820分子・686骨格）。**ただし7で494骨格、あと6件** | **境界そのものが見える。**「要求を494に下げる」「ライブラリを広げる」「2段緩める」の3択 |

**低く置きすぎると「成立しました」しか返りません。**187骨格ある場所に120を頼んでも、
余裕がどれだけあるかは出ますが、何が制約かは出ません。

**境目に置くと、制約の構造が返ります。**500を頼んだときの「回転結合7で494、あと6件」
は、要求を下げるかライブラリを広げるかという判断に直接効きます。686を見せられるより
役に立ちます。

**まず `--dry-run` で読み取りを確認し、試運転を -P（17,775件）で回して境目を探し、
本番を -O（636,669件）で回す**のが実際の使い方です。

**-P の全段で 5.8〜20分**（実測、逐次実行）。掃引そのものは 17,775件を23〜45秒で
読み終わりますが、そこに骨格の正確な数え直し（前線1仕様につきライブラリ1パス）と
順位付けの測定（§9、約4分）と適用範囲の測定が乗ります。**「-P なら1分」は掃引だけの
時間で、答えが返るまでの時間ではありません。**`--no-screenable --no-measure` を付け
れば掃引と数え直しだけになります。

### スライスを変えるだけで答えが変わる

条件を1つも変えず、**ライブラリのスライスだけ**を変えた実測（要求120骨格、一級アミン
strict、`--limit 20000`）。

> この 20,000 は**読み込みの打ち切り**で、§9 の「20,000件までは全件測る」という
> **順位付け測定の上限**とは別の量です。同じ数字が偶然並んでいます。

| スライス | 結果 |
|---|---|
| -N（中性） | **上限39骨格。不成立** |
| -O（+1） | **119骨格。**MW を600に緩めて**124骨格で成立** |

**同じ要求が、スライスの選択だけで不成立から成立に変わります。**一級アミンの含有率が
-N 0.21% / -O 2.46% / -P 13.06%（62倍）だからです。**スライスを書き忘れると -N が選ばれ、
最も不利な場所で測ることになります**（そのため矛盾停止が入っています）。

---

## 2. ライブラリ（スライスを必ず書く）

ZINC22 は tranche を **pH 7.4 の予測プロトン化状態**で分けている。
**描画上の形式電荷ではない。**どのスライスを読むかで答えが変わる。

| 書き方 | スライス | 件数 | 備考 |
|---|---|---|---|
| `cationic ZINC` / `protonated ZINC` / `ZINC O slice` / `+1 ZINC` | **-O（+1）** | 636,669（全件） | **推奨。**一級アミンが -N の 12倍 |
| `ZINC +2 slice` / `dicationic ZINC` / `multiply charged ZINC` / `ZINC P slice` | -P（+2以上） | 17,775（全件） | 濃度は最高（62倍）だが小さい。**試運転向き** |
| `neutral ZINC` / `ZINC N slice` / **`ZINC`（既定）** | -N（中性） | **4,572,322 のうち 1,000,000 のみ取り込み済み** | **矛盾で停止する**（下記） |
| `CHAMP` | 透過性 | 1,132 | |
| `IMI` | 活性 | 22,051 | |
| `GramNegAccum` / `GNA` | 透過性 | 197 | |
| `B3DB` / `blood-brain` / `BBB` | 血液脳関門 | 7,805 | **eNTRy 条件を付けると障壁不一致で停止**（下記） |

**-N だけ打ち切りです。**スライスは 4,572,322 件あり、このディスクには 1,000,000 件
しかありません。画面も `1,000,000 of its 4,572,322 compounds read` と出ます（-O と
-P は `636,669 compounds` のように全件）。**-N の結果はスライス全体についての結果では
ありません。**

### B3DB に eNTRy 条件を付けると止まる

血液脳関門のデータに外膜の条件を向けると、実行前に停止します。

```
about these conditions:
  they are Richter et al., Nature 545:299 (2017), the eNTRy rules, and they
    are about the Gram-negative outer membrane.
  this library is about the blood-brain barrier.
  they are not a general permeability rule, and not a rule about any other
    barrier. In particular not the blood-brain barrier, where the positive
    charge these conditions select for is not what the literature associates
    with penetration

stopping: the conditions asked for are a published rule about a different
  barrier from the one this library measures. Pass --yes to run anyway, having
  read why.
```

**B3DB について答えられること**は別にあります。「自分の集合はこのデータの適用範囲に
入っているか」「何件測れば入るか」は、条件を付けずに聞けば返ります。**screening の
条件を向けるのが止まるだけ**です。

### `ZINC` と書くと `-N` になって止まる

eNTRy の第1条件（イオン化可能窒素）を要求しながら `-N` を読むと矛盾するので、
実行前に停止する。Richter 2017 が**中性 0/39、陰性 0/20、陽性 12/41** と報告しているため。

```
about that library:
  this contradicts the request: the request asks for an ionisable nitrogen,
    which is the first eNTRy condition, but the library is the slice predicted
    to be neutral at pH 7.4
    published evidence: Richter et al. 2017 report 12 of 41 positively charged
      compounds accumulating against 0 of 39 neutral ones
    measured here: on zinc-22x H29P240, primary amines are 0.21% of the -N
      slice, 2.46% of -O and 13.06% of -P
    what to do: screen the -O and -P slices of the same tranche, or say
      explicitly why the neutral slice is wanted

stopping: the library contradicts the published rule the request is built on.
  Pass --yes to run anyway, having read why.
```

承知のうえで流すなら `--yes` を付ける。

---

## 3. 条件（knob）

厳しい順。語彙外は**推測せず停止**する。

### nitrogen

| 書き方 | 設定 | 出自 |
|---|---|---|
| `primary amine` | `primary_amine` | published |
| `secondary amine` / `primary or secondary amine` | `primary_or_secondary_amine` | published |
| `basic nitrogen` / `ionizable nitrogen` / `ionisable nitrogen` | `basic_nitrogen` | published |

3つとも Richter 2017 の「ionisable nitrogen」の範囲内なので published。

### rotatable

| 書き方 | 受け付ける値 |
|---|---|
| `rotatable bonds at most 5` / `rotatable <= 7` / `rotatable bonds ≤ 9` | **5 / 7 / 9 のみ** |

**挿入語は許容する**（2026-10-10 に修正）。`rotatable bonds must be at most 5`、
`the rotatable bonds should be no more than 5` も読める。比較子は語（`at most`
`no more than` `under` `up to` `less than`）でも記号（`<` `<=` `≤` `=`）でもよい。

**`=` も「以下」と読みます。**`rotatable bonds = 7` は「ちょうど7」ではなく
`rotatable_le_7`、つまり**7以下**です。この道具に「ちょうどN」を要求する書き方は
ありません。

5 が published（eNTRy 第3条件）、7 と 9 は relaxation（閾値を動かしただけ）。

**6 と書くと止まる。**丸めない。

### weight

| 書き方 | 受け付ける値 |
|---|---|
| `molecular weight under 450` / `MW under 600` | **450 / 600 のみ** |

- `mw_under_600` は **published（二次経由：Richter 2017 が O'Shea & Moser 2008 を逐語引用）**
- **`mw_under_450` は本研究が選んだ値で、公表された根拠はありません**（画面では
  `chosen by this study; no published bound`）

> registry はこれを `relaxation` に分類していますが、**450 は 600 より厳しい**ので
> 「緩和」とは読まないでください。rotatable の 7・9 は本当に緩い方向への移動で、
> 450 は「公表値が存在しない場所を自分で埋めた」という意味です。**同じ分類語が2つの
> 意味を持っています。**画面は出典の文で書き分けています。

**挿入語は許容する。**`molecular weight is strictly under 450`、`MW must be under 600`、
`mol. wt. 450` も読める。

**500 と書くと止まる。**

> **注意：MW は透過の knob ではない。**Richter 2017 自身が
> 「there is also no correlation between accumulation and molecular weight」
> と書いている。実測でも 450→600 の緩和は 1.1倍しか効かない（回転結合は 2.8倍）。
> サイズ・薬らしさの制約として使うこと。

### charge — **引退しました（2026-10-11）。書くと停止します**

電荷の要求は **knob ではなくライブラリのスライスで書きます。**`neutral` `any charge`
`zero formal charge` `cationic` などを条件として書くと、丸めずに停止します。

```
refused rather than rounded:
  "neutral": charge is chosen by the library slice, not by a requirement. The
    retired knob read the formal charge of the structure as drawn, which is a
    different quantity from the protonation state ZINC predicts at pH 7.4
    registered slices: -N (predicted neutral at pH 7.4), -O (predicted +1 at
      pH 7.4), -M (predicted -1 at pH 7.4) or -P (predicted +2 or more at pH
      7.4)
    name the slice as part of the library instead, for example "cationic ZINC"
      for -O

stopping: the request named something this cannot read as a requirement.
  Nothing was rounded and nothing was redirected to a different field.
```

**`neutral ZINC` や `cationic ZINC` は従来どおり通ります。**ライブラリ句として先に
読まれるので、スライス名として機能します。停止するのは条件として独立に書いた場合です。

#### 引退の理由は inert だからではない

| | |
|---|---|
| **理由** | knob が別の量を測っていた。ZINC は**予測プロトン化状態**で分け、knob は**描画上の形式電荷**を読んでいた |
| **決定的な観測** | `-M`（−1 と予測される群）の標本に、**負の形式電荷で描かれた分子が1件も無かった。**knob は陰イオン集団に到達できない |
| **理由でないこと** | inert だから、ではない。`-O` の標本 20,000 件中 **23件は +1 で描かれている**ので、設定は実際に分子を落とす。落ちたことが見えるかは run の生存数の性質で、knob の性質ではない |

`py/knobs.py` の `RETIRED['charge']` と `BLOCKED['library_charge_slice']` に根拠が
入っています。測定は `py/charge_slices.py` で再現できます。

---

## 4. 譲れない条件には印を付ける

**付けないと全部「緩和してよい」になる。**

```
molecular weight strictly under 450      → strict
MW must be under 450                     → strict
keep rotatable bonds at most 5           → strict
rotatable bonds at most 5                → 緩和してよい
```

認識する語：`strictly` `must` `hard` `required` `no more than`
`non-negotiable` `exactly` `fixed` `do not relax` `keep`

**節単位で効く。**前後40文字、カンマで区切られる。

```
"MW strictly under 450, primary amine"
   → weight は strict、nitrogen は緩和可（隣に漏れない）
```

### 文頭の `must` は全要件に掛かる

```
"The screening must satisfy the following criteria: ... primary amine ...
 rotatable bonds must be at most 5 ..."
```

これは**全 knob が strict** になる。比較する仕様が1つだけになり、**緩和の提案が
出ない**（満たせなければ「不可」で終わる）。

**緩和の提案が欲しいなら、譲れない要件だけに印を付ける。**

---

## 5. 要件にならないもの

2種類あり、扱いが違います。**事実の主張は破棄され、順位付けの要求は答えが返ります。**

### データについての事実や数値 — 破棄される

**入力は要求だけを受け取る。**

| 書くと | 扱い |
|---|---|
| `the 100-million-item cache` | ライブラリ件数の主張 → **破棄。**件数は測定側が出す |
| `the 0.071 AUC deficit` | AUC の主張 → **破棄** |
| `7,394 distinct diversity domains` | 骨格数の主張 → **破棄** |
| `pre-packed` `verified` `confirmed` `proven` `established` | 検証状態の主張 → **破棄** |

破棄したものは理由付きで一覧表示される。**自分の誤りが見える仕組み。**


実例（2026-10-09 に実際に止まったもの）：

```
not taken as input (claims about the data):
  "100-million-item" reads as a library size
    a claim about the data, not a request of it.
  "0.071 auc" reads as an AUC value
  "7,394 distinct diversity domains" reads as a domain count
  "verified" reads as a verification status
```

このとき実際のライブラリは **1,000,000件**（1億ではない）、AUC 差は **0.0540**（0.071 ではない）だった。

### 順位付けや「上位N件」は、要件にならず、答えは返ります（2026-10-13）

`Rank them and give me the top 50` のような文は **knob を設定しません。**この
ツールは分子を出さないので、要件としては読まれず、従来どおり「要件として読まな
かった部分」に出ます。

変わったのは、**その要求に対して順位付けの判定を言うようになったこと**です。

```
you asked for the top 50. It cannot be drawn from this set: the scores are
  too tied for the order to be the model's, so whatever broke the tie would
  be choosing them. The verdict above is about this request.
```

以前は、要求が「読まなかった部分」に落ち、30行下に「この集合は順位付けできない」
という判定が別に出て、**両者が同じ話だと誰も言いませんでした。**判定は画面と
ANSWER ブロックの両方で、要求に対して述べられます。

順位付けが**できる**と出た場合も、分子は出ません。判定が許可するのはスクリーンを
回すことで、このツールがリストを渡すことではありません。


---

## 6. 回せない knob

書いても `not_answerable` が返る。**どの設定にしても解けない。**

| knob | なぜ | 解除に必要なもの |
|---|---|---|
| **標的** | **標的を記録したラベルが1つも無い。**全データが二値 active/inactive | 化合物ごとに標的が記録されたアッセイ |
| globularity | eNTRy 第2条件。3D 配座が必要 | 配座生成器。**実装済みだがこのライブラリでは 100% 通過して無効** |
| スライス | 仕様の `library` で選ぶもので knob ではない | — |

### 標的の書き方（2026-10-12 に変わりました）

**標的は「位置」で読みます。**`-ase` で終わる語だけでは標的になりません。

| 書き方 | 読まれ方 |
|---|---|
| `an inhibitor of transglycosylase` | **標的として照合する** |
| `a transglycosylase inhibitor` | **標的として照合する** |
| `against transglycosylase` | **標的として照合する** |
| `transglycosylase` だけ | 候補として記録し、証拠側にその語を含むラベルがあれば昇格。無ければ報告して走り続ける |

以前は `-ase` で終わる語を見つけた時点で標的に昇格させていました。英語には
`purchase` `increase` `phrase` `database` `case` のような普通の名詞が同じ形で
あり、**`purchase` が標的に昇格して、構造条件を評価する前に走行が止まりました。**
除外語リストを伸ばすのは直し方ではありません（次の普通の語も載っていない）。

昇格しなかった語はこう出ます。

```
looks like a target, and was not taken as one:
  "purchase": it ends in -ase, which is the shape of an enzyme name, but no
    label in the evidence mentions it and nothing in the request puts it where
    a target goes
    if it is the target, write it as "an inhibitor of purchase". Ordinary
      English words end this way too, and one of them stopped a run before
      anything was screened.
```

**既知の制限：**接尾辞の前に4文字を要求するので、`lipase` `urease` のような
短い実在の酵素名は候補にもなりません。**読み違えるのではなく、見落とします。**
`an inhibitor of lipase` と書けば読みます。

```
targets named in the request:
  a label for transglycosylase: cannot be answered
    no label column or value mentions transglycosylase. A model fitted here
    predicts whatever its binary label recorded, which is not this.
```

---

## 7. 所要時間

掃引は **27仕様**（3 knob × 3設定）。**charge の引退前は 54仕様だったので、それより
前の記録に出てくる「54仕様」はその時点の registry のことです。**

| ライブラリ | 掃引（54仕様で実測） |
|---|---|
| -P（17,775件） | **約1分**。試運転向き |
| -O（636,669件） | **約20分** |
| -N（1,000,000件） | 約40分 |

**上の時間は 54仕様のときの実測です。**27仕様で測り直していないので、置き換えずに
出典のまま残しています。1パスなので仕様数は所要時間を支配しませんが、**測っていない
数字を書かないために据え置きます。**

### フラグ

| | |
|---|---|
| `--yes` | 読む前の確認を飛ばす。矛盾の警告を読んだうえで流すときにも使う |
| `--dry-run` | 読み取りだけで即時。**何が要件に写ったかを確かめる最短の手段** |
| `--limit N` | N分子で打ち切る |
| `--no-exact` | 前線の正確な数え直しを省く（推定値になり、そう表示される） |
| `--no-screenable` | 画面が何を立証するかの判定を省く |
| `--no-measure` | 適用範囲外のとき、何件測れば閉じるかの計算を省く |
| `--full` | 順位付けを生存数に関わらず全件で測る（下記） |
| `--out-dir DIR` | `request.json` ほかの書き出し先 |
| `--spec FILE` | 散文の代わりに構造化された仕様を読む。**記録・提出用はこちら** |

**`--dry-run` を先に回すのを勧めます。**散文は閉じた語彙への照合なので、要件に
写らなかった句があれば `nothing in the request was read as a requirement from:`
に出ます。それを読んでから本番を回せば、20分待ってから取りこぼしに気づくことが
ありません。

### 順位付けの判定にかかる時間（2026-10-13 実測）

**生存数に依存しません。**費用は学習（22,051行 × 10エポック）で、生存分子の採点は
その中に消えます。

```
n=   200   221.89  220.56  196.36   秒
n=   600   254.78
n=  1200   262.03
n=  1820   242.92  236.77
n= 13946   248.06            ← 70倍読んで同じ時間
```

**これは暖機後の値だけです。**各バッチの1点目はコールドスタートで遅く、除外した2点は
`n=200 273.14` と `n=1820 299.84`。除外した理由は、同じ 1,820 件が1点目で 299.84 秒、
4点目で 242.92 秒になり、**位置が件数より大きく効いていた**からです。選別は妥当ですが、
選別したと書かずに表だけ出すのは、この道具が禁じている形なので明記します。

`196.36` は2つ目のバッチの6点目（13,946件の測定の後）で、最も暖まった状態の値です。

だから標本化しません。生存数が **20,000件まで**は全件測ります。それを超えると
費用が未測定の領域に入るので、判定を保留して所要時間を添えます。`--full` を付け
れば件数を問わず測ります。

```
  ordering the survivors: not settled by what was measured
  ...
      run it again with --full, which measures all N survivors
```

**20,000 の根拠：**測定した範囲の上端は 13,946 件です。20,000 はそこを -P スライス
全体（17,775）が収まるまで**1.4倍に延長した値**で、測定ではありません。それを超える
領域の費用は測っていないので、「遅い」ではなく「未測定」として判定を保留します。
床の `declared, not derived` と同じ扱いです。

---

## 8. 最小の例

```bash
occam "256 distinct scaffolds from the cationic ZINC library, \
       primary amine, rotatable bonds at most 5, \
       molecular weight strictly under 450"
```

```json
{
  "library": "data/work/zinc_O",
  "deliverable": { "distinct_scaffolds": 256 },
  "requirements": [
    { "knob": "nitrogen",  "setting": "primary_amine",    "strict": false },
    { "knob": "rotatable", "setting": "rotatable_le_5",   "strict": false },
    { "knob": "weight",    "setting": "mw_under_450",     "strict": true  }
  ]
}
```

---

## 9. 返ってくるもの

**画面と記録は語彙が違う**（2026-10-10 に分離、`py/render.py`）。画面には内部識別子を
出さない。`data/work/*.json` には識別子のまま残す。後から解釈なしに読めるのが記録の
存在理由なので、そちらは変えていない。

| 画面 | `request.json` ほか |
|---|---|
| `under 450 Da` | `mw_under_450` |
| `an unhindered primary amine` | `primary_amine` |
| `ZINC22 tranche H29P240, -O slice (+1 at pH 7.4), 636,669 compounds` | `data/work/zinc_O` |
| `cannot be answered` | `not_answerable` |
| `not settled by what was measured` | `not_settled_at_this_sample` |
| `ZINC22 tranche H29P240, -N slice (neutral at pH 7.4), 1,000,000 compounds` | **`data/work/zinc`** |

**記録では -N だけパスに文字が付きません。**`-O` は `zinc_O`、`-P` は `zinc_P`、
`-N` は `zinc`。画面には `-N slice (neutral at pH 7.4)` と出るので読み違えませんが、
`data/work/*.json` を直接読むときは、**接尾辞の無い `zinc` が中性スライス**で、
公表された規則が「集積しない」と言っている側であることに注意してください。

画面に出る見出しは以下。**順序もこの通り**で、どれも出ないことがあります。

```
read as:                                 読み取った内容（必ず表示）
  library / deliverable                  ライブラリと成果物
  <性質> <要求>  (not negotiable)        要求・出自の論文・譲れるか・knob の注意
not taken as input (claims about the data):
                                         破棄した事実の主張
refused rather than rounded:             語彙外の値と、有効な候補
nothing in the request was read as a requirement from:
                                         要件に写らなかった句  ← 必ず読む
  （機能語だけの断片は件数だけ出る。"every part of the request was read." と
    出たら取りこぼしは無い）
looks like a target, and was not taken as one:
                                         -ase の形だが標的にしなかった語
targets named in the request:            標的の照合結果
about that library:                      スライスと矛盾判定
about these conditions:                  条件がどの障壁についてか（別障壁のとき）
what to screen instead (measured, on this disk):
                                         止まったときに読めるスライス
comparing N specifications …:            比較した仕様数と1パスの実測
requirements that did not change the result:
                                         何も絞らなかった要求（旧 [INERT]）
verdict:                                 可否
as asked:                                要求どおりの仕様の結果
the request as asked: / recommended instead:
                                         緩和が無ければ前者。緩和した要求とその出自
closest specification that did not reach:
                                         正確に数えて届かなかった最も近い仕様と、
                                         あと何件か  ← 要求を下げるか広げるかの判断
what would unblock it:                   届かないときに必要なもの
what a screen of this specification would establish:
                                         順位付けの可否・適用範囲・学習ラベル
  the N% floor is declared, not derived… 床に出典が無いことの開示
  this rests on …, not proven            標本測定のとき、何に依拠しているか
  positive control: passes               **対照。重い判定には必ず付く**
which measured set is nearest …:         代替案（実測のみ。化学の提案はしない）
you asked for the top N. …               順位付けを要求したときだけ。判定をその
                                         要求に対して述べる
========== ANSWER ==========             **最後に全部を1ブロックで。**
                                         可否 → 画面が何を立証するか → 次に何を測るか
```

**ANSWER ブロックだけ読めば決められるように作ってあります。**上の詳細はその数字が
どこから来たかで、答えを検算するためのものです。

`data/work/` に `request.json` / `sweep.json` / `proposal.json` / `screenable.json` が残る。

---

## 10. 散文の限界（提出物には `--spec` を使う）

散文パーサは**閉じた語彙への決定的な照合**で、LLM を使っていない。だから：

- **語彙外の句は捨てられる。**`nothing in the request was read as a requirement from:`
  を必ず読むこと。要件だったのに捨てられている可能性がある（この見出しは
  §9 の一覧と同じ文字列です。以前ここには短い別の文字列が書かれていて、
  画面には存在しませんでした）
- `effectively penetrate the outer membrane` のような句は**何にも写らない**

**論文・提出物・特許の記録には `--spec request.json` を使う。**
何を要求したかが解釈なしに残る。散文は端末での入口に留める。
