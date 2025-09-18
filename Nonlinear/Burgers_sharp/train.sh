#!/bin/sh
function runTrain(){
    for file in `ls $1`
        do
            if [ -e $1"/"$file"/scale_basis_uniform.py" ]
            then
                echo "start run $file"
                cd $1"/"$file
                nohup python scale_basis_uniform.py > /dev/null 2>&1 &
                wait
                echo "$file done! Exist status: $?"
            fi
        done
}

runPath=$(cd `dirname $0`;pwd)
echo $runPath
runTrain $runPath