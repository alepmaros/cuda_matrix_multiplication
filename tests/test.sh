#!/bin/bash

TIME_GLOBAL=0
TIME_SHARED=0
TIME_GLOBAL_TOTAL=0
TIME_SHARED_TOTAL=0
LINES=1024
INCREMENT=256
MAXLINES=10240
NTESTSPERLINE=7
COUNTER=0

nvcc ../cuda_matrix_global.cu -o ./cg.o -D __NO_OUTPUT -D __TIME -Wno-deprecated-gpu-targets
nvcc ../cuda_matrix_shared.cu -o ./cs.o -D __NO_OUTPUT -D __TIME -Wno-deprecated-gpu-targets

if [ -f ./times.csv ]; then
    rm ./times.csv
fi

echo 'lines;time_global;time_shared' >> times.csv

awk 'BEGIN { printf "%-7s %-20s %-20s\n", "LINES", "TIME_GLOBAL", "TIME_SHARED"}'

while [  $LINES -le $MAXLINES ]; do
    TIME_GLOBAL_TOTAL=0
    TIME_SHARED_TOTAL=0
    COUNTER=0
    while [ $COUNTER -lt $NTESTSPERLINE ]; do
        # Generate a new set of matrices
        python3 ../generate_input.py $LINES $LINES $LINES
        TIME_GLOBAL="$(./cg.o < input)"
        TIME_SHARED="$(./cs.o < input)"
  
        TIME_GLOBAL_TOTAL=$(python -c "print($TIME_GLOBAL_TOTAL+$TIME_GLOBAL)")
        TIME_SHARED_TOTAL=$(python -c "print($TIME_SHARED_TOTAL+$TIME_SHARED)")
          
        let COUNTER=COUNTER+1
    done
   
    TIME_GLOBAL=$(python -c "print($TIME_GLOBAL_TOTAL/$NTESTSPERLINE)")
    TIME_SHARED=$(python -c "print($TIME_SHARED_TOTAL/$NTESTSPERLINE)")
    
    echo $LINES';'$TIME_GLOBAL';'$TIME_SHARED >> times.csv

    awk 'BEGIN { printf "%-7d %-20f %-20f\n", '$LINES', '$TIME_GLOBAL', '$TIME_SHARED'; exit; }'

    let LINES=LINES+INCREMENT
done
