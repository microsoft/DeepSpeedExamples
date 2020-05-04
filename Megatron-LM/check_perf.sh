#!/bin/bash

N=200

cnt=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | wc -l`
ar_cnt=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | grep -i "allreduce" | wc -l`


let pass=$N+5

if [ $cnt -gt $pass ];
then
    fw_avg=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v n="$N" -v sum=0 '{sum += $6} END {print sum/n}'`
    fw_max=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v max=0 '{if ($6 > max) max = $6} END {print max}'`
    bw_avg=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v n="$N" -v sum=0 '{sum += $9} END {print sum/n}'`
    bw_max=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v max=0 '{if ($9 > max) max = $9} END {print max}'`

    if [ $ar_cnt -gt 0 ];
    then
        ar_avg=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v n="$N" -v sum=0 '{sum += $12} END {print sum/n}'`
        ar_max=`cat $1 | grep -i "^worker-0: time (ms) | forward:" | tail -n ${N} | awk -v max=0 '{if ($12 > max) max = $12} END {print max}'`
    else
        ar_avg=0
        ar_max=0
    fi

    ee_avg=`cat $1 | grep -i "^worker-0:  iteration" | tail -n ${N} | awk -v n="$N" -v sum=0 '{sum += $11} END {print sum/n}'`
    ee_max=`cat $1 | grep -i "^worker-0:  iteration" | tail -n ${N} | awk -v max=0 '{if ($11 > max) max = $11} END {print max}'`
    printf "FW_AVG: %.2f, FW_MAX: %.2f, BW_AVG: %.2f, BW_MAX: %.2f, AR_AVG: %.2f, AR_MAX: %.2f, EE_AVG: %.2f, EE_MAX: %.2f *** [$1]\n" $fw_avg $fw_max $bw_avg $bw_max $ar_avg $ar_max $ee_avg $ee_max
else
    printf "No enough result yet *** [$1]\n"
fi
