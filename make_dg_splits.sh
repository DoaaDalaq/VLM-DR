#!/bin/bash
# Build leave-one-domain-out splits. Usage: bash make_dg_splits.sh APTOS
# (target: APTOS, DEEPDR, IDRID, MESSIDOR or RLDR; FGADR is always a source domain)
T=$1; D=splits/dg
: > splits/train_no$T.txt; : > splits/val_no$T.txt
for X in APTOS DEEPDR IDRID MESSIDOR RLDR FGADR; do
  [ "$X" = "$T" ] && continue
  awk 1 $D/${X}_train.txt    >> splits/train_no$T.txt
  awk 1 $D/${X}_crossval.txt >> splits/val_no$T.txt
done
cp $D/${T}_crossval.txt splits/test_$T.txt
echo "train: $(wc -l < splits/train_no$T.txt)  val: $(wc -l < splits/val_no$T.txt)  test: $(wc -l < splits/test_$T.txt)"
