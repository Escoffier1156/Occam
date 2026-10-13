# The six prompts

Records of one machine, one day, one library. `tests/test_six_prompts.py`
reads them. They are not ground truth about chemistry: a test comparing
against them says that the tool still answers what it answered.

Recorded 2026-10-13 at commit `bbed7c1` plus the sampling and reporting
changes that followed it, against `data/work/zinc_P` (ZINC22 tranche H29P240,
-P slice, 17,775 compounds).

## The prompts

1. `120 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450`
2. `Gram-negative accumulation screen, E. coli. All three eNTRy conditions, not two: an unhindered primary amine, globularity at most 0.25, rotatable bonds at most 5. No molecular weight bound unless you can cite one. Give me the scaffold count and state the scaffold definition you used.`
3. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. State the scaffold definition you used.`
4. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. Rank them and give me the top 50 to purchase.`
5. `256 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450. Rank them and give me the top 50.`
6. `500 distinct scaffolds from the dicationic ZINC library, primary amine strictly required, rotatable bonds at most 5, molecular weight under 450.`

Each prompt is also in its run's `request.json` under `goal`, which is where
the test reads it from. This file is the human copy; the records are the
machine's.

## What each one is for

| run | why it is in the set |
| --- | --- |
| 1 | a request the library fills as asked |
| 2 | no library named, and a condition this pipeline cannot evaluate. It must stop, so there is no proposal |
| 3 | one setting loosened |
| 4 | the same, plus a ranking request and `purchase`, a word shaped like an enzyme name |
| 5 | the same, plus a ranking request and no such word |
| 6 | two settings loosened, the largest survivor set of the six, and the only near miss |

Runs 3, 4 and 5 are the same specification. They are kept separate because
their measurements must agree, and the day they did not was the day a sampled
verdict was found.

## Re-recording

Four at a time, longest first; about twenty minutes.

```bash
S=/tmp/occam-six && mkdir -p $S
nix-shell --run '
printf "%s\n" 6 5 4 1 3 2 | xargs -P 4 -I{} sh -c "
  n={}
  mkdir -p '$S'/run\$n
  python3 -u py/occam.py --yes --out-dir '$S'/run\$n \"\$(sed -n \"\${n}p\" prompts.txt)\" > '$S'/run\$n.log 2>&1
"'
for n in 1 2 3 4 5 6; do
  cp $S/run$n/*.json tests/fixtures/six_prompts/run$n/
done
```

Then run the slow layer against them:

```bash
nix-shell --run 'OCCAM_SIX=1 python3 -m unittest tests/test_six_prompts.py'
```

**A re-recording that moves a verdict needs a sentence in the commit saying
which verdict moved and why.** The fixtures exist to make a changed answer
visible; a fixture updated silently hides exactly what it was added to show.

## Verdicts as recorded

Every ordering verdict is measured on every survivor, so none of them rests
on the sampled-upper-bound assumption.

| run | recommended | survivors | scaffolds | ordering | distinct share |
| --- | --- | --- | --- | --- | --- |
| 1 | as asked | 430 | 187 exact | cannot be answered | 0.486 on 430 of 430 |
| 2 | — | — | — | — | stops: no library named |
| 3 | RB≤7 | 1,200 | 475 exact | cannot be answered | 0.4683 on 1,200 of 1,200 |
| 4 | RB≤7 | 1,200 | 475 exact | cannot be answered | 0.4683 on 1,200 of 1,200 |
| 5 | RB≤7 | 1,200 | 475 exact | cannot be answered | 0.4683 on 1,200 of 1,200 |
| 6 | RB≤9 | 1,820 | 686 exact | cannot be answered | 0.4648 on 1,820 of 1,820 |

Run 6's nearest miss: RB≤7 / MW<600, 1,292 molecules, 494 distinct scaffolds
counted exactly, six short of the 500 asked for.

Run 6 read `answerable` at 0.5967 before these records, measured on 1,200 of
its 1,820 survivors. The share falls as more are measured, so that pass was
the sample. It is the verdict that moved, and this is the sentence.
