#!/bin/sh
function runTrain(){
    for file in `ls $1`
        do
            if [ -e $1"/"$file"/train.sh" ]
            then
                echo "start run $file"
                cd $1"/"$file
                sh train.sh
                wait
                echo "$file done! Exist status: $?"
            fi
        done
}

runPath=$(cd `dirname $0`;pwd)
echo $runPath
runTrain $runPath