#!/bin/bash
for ((a=0; a <= 1; a++)) #a 为--stratified，确定是否采用分层采样。a =0 或者 a= 1
do
  for ((b=1; b<= 8; b++)) # b 为train_set=test_num：1-7
  do
    python mf.py -t $b -s $a
  done
done

# 就是循环了14次，同构的时候a=1，从1-7迭代一遍，异构的时候a=0,从1-7迭代了一次。