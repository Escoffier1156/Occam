# Occam へのプロンプト：形式と注意点

2026-10-10。内容は `py/occam.py` と `py/knobs.py` の実装から転記。
**語彙を足したらこのファイルも直す。**ずれたら実装が正しい。

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
| **条件** | 下の knob から | 既定（最も厳しい設定）になる |
| **譲れるか** | `strictly` などを添える | **すべて「緩和してよい」になる** |

### 成果物は「何個の何か」を書く

```
256 distinct scaffolds     ← 可
256 unique candidates      ← 可
256                        ← 不可。何の256個か分からない
```

**判定基準になるのは scaffold の数**で、生存数ではない。成果物が「1骨格1代表」なので、
制約は骨格の数に掛かる。

---

## 2. ライブラリ（スライスを必ず書く）

ZINC22 は tranche を **pH 7.4 の予測プロトン化状態**で分けている。
**描画上の形式電荷ではない。**どのスライスを読むかで答えが変わる。

| 書き方 | スライス | 件数 | 備考 |
|---|---|---|---|
| `cationic ZINC` / `protonated ZINC` / `ZINC O slice` | **-O（+1）** | 636,669 | **推奨。**一級アミンが -N の 12倍 |
| `ZINC +2 slice` / `dicationic ZINC` | -P（+2以上） | 17,775 | 濃度は最高（62倍）だが小さい。**試運転向き** |
| `neutral ZINC` / **`ZINC`（既定）** | -N（中性） | 1,000,000 | **矛盾で停止する**（下記） |
| `CHAMP` | 透過性 | 1,132 | |
| `IMI` | 活性 | 22,051 | |
| `GramNegAccum` / `GNA` | 透過性 | 197 | |

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

5 が published（eNTRy 第3条件）、7 と 9 は relaxation（閾値を動かしただけ）。

**6 と書くと止まる。**丸めない。

### weight

| 書き方 | 受け付ける値 |
|---|---|
| `molecular weight under 450` / `MW under 600` | **450 / 600 のみ** |

- `mw_under_600` は **published（二次経由：Richter 2017 が O'Shea & Moser 2008 を逐語引用）**
- `mw_under_450` は relaxation（本研究の選択）

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

## 5. 書いてはいけないもの

**データについての事実や数値。**入力は要求だけを受け取る。

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

---

## 6. 回せない knob

書いても `not_answerable` が返る。**どの設定にしても解けない。**

| knob | なぜ | 解除に必要なもの |
|---|---|---|
| **標的** | **標的を記録したラベルが1つも無い。**全データが二値 active/inactive | 化合物ごとに標的が記録されたアッセイ |
| globularity | eNTRy 第2条件。3D 配座が必要 | 配座生成器。**実装済みだがこのライブラリでは 100% 通過して無効** |
| スライス | 仕様の `library` で選ぶもので knob ではない | — |

標的名（`-ase` で終わる語、`inhibitor of X`）は自動検出して照合する。

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

`--limit N` で打ち切れる。**`--dry-run` なら読み取りだけで即時。**

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

画面に出る見出しは以下。

```
read as:                               読み取った内容（必ず表示）
  library / deliverable                ライブラリと成果物
  <性質> <要求>  (not negotiable)      要求・出自の論文・譲れるか・knob の注意
not taken as input (claims about …):   破棄した事実の主張
refused rather than rounded:           語彙外の値と、有効な候補
nothing in the request was read …:     要件に写らなかった句  ← 必ず読む
  （機能語だけの断片は件数だけ出る。"every part of the request was read." と
    出たら取りこぼしは無い）
targets named in the request:          標的の照合結果
about that library:                    スライスと矛盾判定
comparing N specifications …:          比較した仕様数と1パスの実測
requirements that did not change …:    何も絞らなかった要求（旧 [INERT]）
verdict:                               可否
the request as asked: / recommended instead:
                                       緩和が無ければ前者。緩和した要求とその出自
what would unblock it:                 届かないときに必要なもの
what a screen … would establish:       順位付けの可否・適用範囲・学習ラベル
  positive control: passes             **対照。重い判定には必ず付く**
what could be done instead (measured…): 代替案（実測のみ。化学の提案はしない）
```

`data/work/` に `request.json` / `sweep.json` / `proposal.json` / `screenable.json` が残る。

---

## 10. 散文の限界（提出物には `--spec` を使う）

散文パーサは**閉じた語彙への決定的な照合**で、LLM を使っていない。だから：

- **語彙外の句は捨てられる。**`nothing was read from:` を必ず読むこと。
  要件だったのに捨てられている可能性がある
- `effectively penetrate the outer membrane` のような句は**何にも写らない**

**論文・提出物・特許の記録には `--spec request.json` を使う。**
何を要求したかが解釈なしに残る。散文は端末での入口に留める。
