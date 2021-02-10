python sharpness-generalization.py $f 2>/dev/null

python collect-volume.py 2>/dev/null
echo 'Plotting figures...'
python collect-plot.py 2>/dev/null

