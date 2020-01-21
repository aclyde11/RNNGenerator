# RNN Generator

Based 99.98\% on the model from [1]


# How to use Molecular Generator Code


## Getting the code.

You need to set up a conda env. This will look different for different machines. 

```python
conda create -n myenv python=3.6
conda activate my env

###
#This step is very machine specific... see https://pytorch.org/get-started/locally/
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
###

pip install gensim matplotlib pandas tqdm 
conda install -c rdkit rdkit 
```

## Getting started with the first model. 
1. Download a seed dataset to start training (should be larger than 1M). The dataset should have just a single column for smiles.

```
OCCCNc1cc(-c2ncnc(Nc3cccc(Cl)c3)n2)ccn1 m_0
O=c1[nH]c2ccccc2n1C1CCN(Cc2ccc(-c3nc4cc5[nH]cnc5cc4nc3-c3ccccc3)cc2)CC1 
Oc1nc2ccccc2n1C1CCN(Cc2ccc(-c3nc4ccccc4nc3-c3ccccc3)cc2)CC1 
Cc1nc(-c2ccccc2)c(-c2ccc(CN3CCC(n4c(=O)[nH]c5ccccc54)CC3)cc2)[nH]c1=O 
O=c1[nH]c(-c2ccc(CN3CCC(n4c(=O)[nH]c5ccccc54)CC3)cc2)c(-c2ccccc2)nc1Cc1ccccc1 
O=c1[nH]c(-c2ccc(CN3CCC(n4c(=O)[nH]c5ccccc54)CC3)cc2)c(-c2ccccc2)nc1Cc1ccc(O)cc1 
CCC(C)c1nc(-c2ccccc2)c(-c2ccc(CN3CCC(n4c(=O)[nH]c5ccccc54)CC3)cc2)[nH]c1=O
```

2. Build init vocabulary. This turns charaters such as "C" or "(" into integer numbers. These integers numbers are then feed into an embedding layer which turns 43 -> [0.23, 0.432, 0.343] etc. In effect we have a three way map:

```python
i2c = {23 : 'C', 34 : '@', ...}
c2i = {'C' : 23, '@' : 34, ...}
embedding = {23 : [0.234, ..., 456.32], 34 : [34.34, ..., 0.3432]}
```

This can be done by 
```shell
emacs config.py #verify this is correct for your run. Must stay constant for entire exp. 
python model/vocab.py -i mosesrun/smiles.txt -o mosesrun --maxlen 318 --permute_smiles 5 --start -p 8 # use eight threads
```
take note the parameters maxlen as it must be kept constant. In general we set vocab size to something a bit bigger than this script reports just in case we encouter some strange variable down the road. Also take that with the permute_smiles options we turned 1.5M Chembl smiles into 10M smiles by rearranging the smiles. This may take awhile, and start says to rebuild vocab.

3. Train Initial Model
The initial model training takes an untrained model and trains it to be as good as possible on the original dataset. This must be done on a GPU otherwise it will take forever. 

```shell
python train.py -i mosesrun/ -b 1024 --logdir mosesrun/ -e 10 
```

Now the model is contained in the folder. To continue training 

```
python train.py -i mosesrun/ -b 1024 --logdir mosesrun/ -e 10  --ct
```

## Get some samples
```shell
python train.py -i mosesrun/ -n 100 --vr -o samples.txt
```

## Fine Tuning Steps

Fine tuning is attempting to shift the properties of the model towards something desired. For instance, you can sample the generator, compute SA scores for 10k compounds, and retrain the generator on the top 1k for SA scores. Then when you resample the generator, the distribution of SA scores should have shifted. 


1. Sample Compounds 
```shell
python infer.py -i mosesrun/ --logdir mosesrun/ -o first_samples.txt -n 10000 -vr 
```
This will use the model from chemblrun to create 10,000 samples and validate the samples. 

2. Score and create subset
The codes here for scoring molecules is not ready yet. You can do this on your own somehow. 
```shell
mkdir round1
head -n 1000 first_samples.txt > round1/topscores.smi
cp mosesrun/model.autosave.pt round1/.
cp mosesrun/vocab.txt  round1/.
```

Now we need to create the encoding vocab set. This will create a bunch of random permutation of the provided smiles strings (in this attempt to make 100 for each), and will create the proper data files for training. 
```shell
python model/vocab.pt -i round1/topscores.smi -o round1/ -n 8 --maxlen 318 --permute_smiles 100 
```

4. Retrain
using --ct flag to continue from the model in the folder -i 
```shell
python train.py -i round1 -b 1024 --logidr round1 -e 5 --ct
```


Refereces:
1. Gupta, A., Müller, A., Huisman, B., Fuchs, J., Schneider, P., Schneider, G. (2018). Generative Recurrent Networks for De Novo Drug Design Molecular Informatics  37(1-2)https://dx.doi.org/10.1002/minf.201700111
2. Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Nikolenko, S., Aspuru-Guzik, A., Zhavoronkov, A. (2018). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Modelshttps://arxiv.org/abs/1811.12823

