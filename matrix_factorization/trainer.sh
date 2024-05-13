#!/bin/bash
for ((a=0; a <= 1; a++)) 
do
  for ((b=1; b<= 8; b++))
    python mf.py -t $b -s $a
  done
done
