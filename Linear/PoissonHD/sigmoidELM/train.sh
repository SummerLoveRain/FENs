#!/bin/sh
N_start=10
N_end=50
N=10
let "N_now=$N_start"
while [ $N_now -le $N_end ]
do 
    echo "start N_basis=$N_now."
    nohup python train_sh.py cuda $N_now > /dev/null 2>&1
    wait
    echo "N_basis=$N_now done! Exist status: $?"
    let "N_now=$N_now+$N"
done