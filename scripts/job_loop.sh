#!/bin/bash

#nohup nice -n 10 python data_creation_pipeline.py &> logs/datagen_LH0-5.out &

#declare -a vals=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#declare -a vals=('1.38' '1.94' '2.75' '3.89' '5.50' '7.78' '11.00' '15.56' '22.00')
declare -a vals=('0.000625' '0.000884' '0.001250' '0.001768' '0.002500' '0.003536'
 '0.005000' '0.007071' '0.010000')
for val in "${vals[@]}"
do
    #tag_lgal="_FracZSNIItoHot${val}"
    #tag_lgal="_FeedbackEjectionEfficiency${val}"
    tag_lgal="_AgnEfficiency${val}"
    echo "nohup nice -n 10 data_creation_pipeline.py &> logs/datagen_LH${idx_LH}.out &"
    nohup nice -n 10 data_creation_pipeline.py idx_LH &> logs/datagen_LH${idx_LH}.out &
done
