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
you can simply comment out line 196-197 and uncomment line 199 in `sharpness-generalization.py`.

For CIFAR-10 dataset, you need to load the data from `data/x_train_car_and_cat.npy` etc. at line 328-331, and also modify the training set size at line 335 and input dimension for the input layer of neural network at line 152. We leave these tunability so readers can customize their own experiments conveniently. Also our code has necessary in-place comments so readability is guaranteed.

For Entropy-SGD implementation, we use code from [here](https://github.com/Justin-Tan/entropy-sgd-tf) (which is not related to authors),
please refer to that repository if motivated.
