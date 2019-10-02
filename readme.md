# RNN Generator

Based 100\% on the model from [1]


## Running
I include a few pretrained models:
Right now, verifying output requires -v and needs RdKit. Will remove soon.
1. MOSES [2]
```shell script
python infer.py -i mosesrun/ --logdir mosesrun/ -v -o output.txt -n <num_to_gen> -t <sample_temp> \
      [--batch_size batch_size]
```

## Training 
1. First you need to make a vocab for your smiles list. Please format all smiles in txt file with one smile per line.
```shell script
python model/vocab.py -i <smiles.txt> -o <proc_data_dir/> --maxlen <number>
```

Make sure you keep track of the vocab size and the max length you chose. Update the config in the train.py!!!

2. Now you can train the model.  

```shell script
python train.py --input <proc_data_dir/> --logdir <logdir/> 
```

Refereces:
1. Gupta, A., Müller, A., Huisman, B., Fuchs, J., Schneider, P., Schneider, G. (2018). Generative Recurrent Networks for De Novo Drug Design Molecular Informatics  37(1-2)https://dx.doi.org/10.1002/minf.201700111
2. Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Nikolenko, S., Aspuru-Guzik, A., Zhavoronkov, A. (2018). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Modelshttps://arxiv.org/abs/1811.12823

