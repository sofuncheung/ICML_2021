# Example code for paper *WHY FLATNESS CORRELATES WITH GENERALIZATION FOR DEEP NEURAL NETWORKS*

For all of our experiments in the main text with open datasets, we provide source code to reproduce the results.
This includes figure. 3 to figure. 5 in the main paper, and various results in the Appendix, with minimal modification to the code.

## Environment:

Platform: Linux

GPU (tested on): NVIDIA GeForce RTX 2080 Ti

CPU (tested on): Our anonymous computing cluster 

We recommend using you own python3 virtual environment.

Just create and **activate** your own virtual environment, 

```
mkdir ~/.venv 2>dev/null
pushd ~/.venv
python3 -m venv --system-site-packages MyEnvName
# alter the VE's name as you like
cd MyEnvName
source bin/activate
popd
python3 -m pip install --upgrade pip setuptools wheel
```

then install the dependency specified by `requirements.txt`:

`python3 -m pip install -r requirements.txt`.

Then you are all set!

1. Reproducing figure. 3 (a), (b), (d), (e) and figure. 4, see [FCN/](https://anonymous.4open.science/repository/c12272d1-2823-453b-b59b-2e9a8905e2f7/FCN/)

2. Reproducing figure. 3 (c), (f), see [ResNet50/](https://anonymous.4open.science/repository/c12272d1-2823-453b-b59b-2e9a8905e2f7/ResNet50/)

3. Reproducing figure. 5, see [Temporal/](https://anonymous.4open.science/repository/c12272d1-2823-453b-b59b-2e9a8905e2f7/Temporal/)

1 and 3 can be run on single CPU in short time, while 2 preferably be run on GPU. 



