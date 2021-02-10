for f in `seq 0 50 500`
do
    mkdir attack_size_$f
    cp sharpness-generalization.py attack_size_$f/
    cd attack_size_$f
    python sharpness-generalization.py $f 2>/dev/null
    cd ..
done

python collect-volume.py 2>/dev/null
echo 'Plotting figures...'
python collect-plot.py 2>/dev/null

