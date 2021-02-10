# Example code for FCN

Usage: activate your virtual environment and make sure all packages in `ICML_2021_code/requirements.txt` have been installed.
Then just simply run:
(if you are using small dataset size like 500 MNIST images for training set, this can be done on a single cpu within 15 minutes)

`./run.sh`

These code will automatically sample different attack sets, train neural nets, calculate sharpness, calculate volume, and finally plot it out for you.

At initial state, these code will reproduce Figure.3 (a) and (d). 
You can conveniently change dataset/optimizer by modifing a few lines in `sharpness-generalization.py`
and `collect-volume.py`. 
These should be very straightforward to see if you read the code carefully. 

For example, if you want to change optimizer from SGD to Adam in order to reproduce Figure. 4 (b) and (e),
you can simply comment out line 198-199 and uncomment line 201 in `sharpness-generalization.py`.

For Entropy-SGD implementation, we use code from https://github.com/Justin-Tan/entropy-sgd-tf (which is not related to authors),
please refer to that repository if motivated.
