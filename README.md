# Example code for paper *WHY FLATNESS CORRELATES WITH GENERALIZATION FOR DEEPNEURAL NETWORKS*

For all of our experiments in the main text with open datasets, we provide source code to reproduce the results.
This includes figure. 3 to figure. 5 in the main paper, and various results in the Appendix, with minimum modification to the code.

## Environment:

Platform: Linux
GPU: NVIDIA GeForce RTX 2080 Ti
CPU: Our anonymous computing cluster 

We recommend using you own python3 virtual environment.
Just create and **activate** your own virtual environment, then run:
`python3 -m pip install -r requirements.txt`.
then you are all set!

1. Reproducing figure. 3 (a), (b), (d), (e) and figure. 4, see FCN/

2. Reproducing figure. 3 (c), (f), see ResNet50/

3. Reproducing figure. 5, see Temporal/

1 and 3 can be run on single CPU in short time, while 2 preferably be run on GPU. 



