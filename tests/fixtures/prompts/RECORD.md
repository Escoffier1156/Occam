# The prompts

Records of one machine against the libraries on it. `tests/test_prompts.py`
reads them. They are not ground truth about chemistry: a test comparing
against them says that the tool still answers what it answered.

**The count is not written down anywhere that matters.** This directory was
called `six_prompts` and the module `test_six_prompts.py`, and both names went
wrong the moment a seventh was added. The set is whatever `run*/` directories
are here; `WHY` in the test module says what each one is for, and a directory
without an entry there, or an entry without a directory, fails.

## The prompts

Each prompt is in its run's `request.json` under `goal`, which is where the
test reads it. The list below is the human copy.

1. `120 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450`
2. `Gram-negative accumulation screen, E. coli. All three eNTRy conditions, not two: an unhindered primary amine, globularity at most 0.25, rotatable bonds at most 5. No molecular weight bound unless you can cite one. Give me the scaffold count and state the scaffold definition you used.`
3. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. State the scaffold definition you used.`
4. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. Rank them and give me the top 50 to purchase.`
5. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. Rank them and give me the top 50.`
6. `500 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450.`
7. `120 distinct scaffolds from the CHAMP library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450.`

## Why each one is here

| run | library | why it is in the set |
| --- | --- | --- |
| 1 | ZINC -P | a request the library fills as asked |
| 2 | — | no library named, and a condition this pipeline cannot evaluate. It must stop, so there is no proposal |
| 3 | ZINC -P | one setting loosened |
| 4 | ZINC -P | the same, plus a ranking request and `purchase`, a word shaped like an enzyme name |
| 5 | ZINC -P | the same, plus a ranking request and no such word |
| 6 | ZINC -P | two settings loosened, the largest survivor set here, and the only near miss |
| 7 | CHAMP | a library held in one parquet file rather than a directory of parts, every row of which fails a condition held as asked |

Runs 3, 4 and 5 land on the same specification. They are kept separate
because their measurements must agree, and the day they did not was the day a
sampled verdict was found. The test derives that grouping from the records
rather than naming the members, because the member list and the method name
had already drifted apart.

Run 7 was added after a day on which four of five faults came out of CHAMP:
`iter_library()` globbed `part-*.parquet` inside a directory, which matches
nothing inside a file path and raises nothing, so the run read zero molecules
and answered that no setting of any knob could fill the request. Nothing in
the set was exercising a single-file library.

## Re-recording

Four at a time, longest first; about twenty minutes. The prompt is read out
of each run's own `request.json`, which is the only copy -- the procedure here
used to read a `prompts.txt` that is not in this repository, so it had never
been run as written.

```bash
S=/tmp/occam-prompts && mkdir -p $S
nix-shell --run '
printf "%s\n" 6 5 4 1 3 7 2 | xargs -P 4 -I{} sh -c "
  n=\$1
  mkdir -p '$S'/run\$n
  prompt=\$(python3 -c \"import json;print(json.load(open(\\\"tests/fixtures/prompts/run\$n/request.json\\\"))[\\\"goal\\\"])\")
  python3 -u py/occam.py --yes --out-dir '$S'/run\$n \"\$prompt\" > '$S'/run\$n.log 2>&1
" sh {}'
```

**Diff before copying.** The whole point of the fixtures is that a changed
answer is visible, and a copy made without reading the difference is the one
operation that destroys that.

```bash
for n in 1 2 3 4 5 6 7; do
  diff <(python3 -m json.tool $S/run$n/screenable.json 2>/dev/null) \
       <(python3 -m json.tool tests/fixtures/prompts/run$n/screenable.json 2>/dev/null)
done
for d in $S/run*; do cp $d/*.json tests/fixtures/prompts/$(basename $d)/; done
```

Then run the slow layer against them:

```bash
nix-shell --run 'OCCAM_PROMPTS=1 python3 -m unittest tests/test_prompts.py'
```

**A re-recording that moves a verdict needs a sentence in the commit saying
which verdict moved and why.** The fixtures exist to make a changed answer
visible; a fixture updated silently hides exactly what it was added to show.

## Verdicts as recorded

Recorded 2026-10-14 at commit `500f4f0`. Every ordering verdict is measured on
every survivor, so none of them rests on the sampled-upper-bound assumption.

| run | recommended | survivors | scaffolds | ordering | distinct share |
| --- | --- | --- | --- | --- | --- |
| 1 | rotatable le 5 + mw under 450 | 430 | 187 exact | not_answerable | 0.486 on 430 of 430 |
| 2 | — | — | — | stops: no library named |
| 3 | rotatable le 7 + mw under 450 | 1,200 | 475 exact | not_answerable | 0.4683 on 1,200 of 1,200 |
| 4 | rotatable le 7 + mw under 450 | 1,200 | 475 exact | not_answerable | 0.4683 on 1,200 of 1,200 |
| 5 | rotatable le 7 + mw under 450 | 1,200 | 475 exact | not_answerable | 0.4683 on 1,200 of 1,200 |
| 6 | rotatable le 9 + mw under 450 | 1,820 | 686 exact | not_answerable | 0.4648 on 1,820 of 1,820 |
| 7 | not reachable | 0 | 0 sketch | — | — |

Run 6's nearest miss: RB<=7 / MW<600, 1,292 molecules, 494 distinct scaffolds
counted exactly, six short of the 500 asked for.

Run 7 reaches zero because CHAMP holds no primary amine: 0 of 1,132, against
71 of 197 in GramNegAccum, 2,756 of 22,051 in IMI-COMBINE and 2,322 of 17,775
in the ZINC -P slice. The flexibility and weight bounds keep 94.7% and 99.7%
of the library on their own, so the run names the amine as the condition that
removed everything, and says it was held as asked so no looser setting of it
was compared.
