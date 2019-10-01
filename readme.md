# RNN Generator

Use 
```shell script
python infer.py -n <number_to_gen> -m <model.pt> -o <output.smi> -t <sampling_temperature> \
    [-g <gpu number>] 
```

### Training

1. First you need to make a vocab for your smiles list. Please format all smiles in txt file with one smile per line.
```shell script
python model/vocab.py -i <smiles.txt> -o <proc_data_dir/> --maxlen <number>
```

Make sure you keep track of the vocab size and the max length you chose. Update the config in the train.py!!!

2. Now you can train the model.  

```shell script
python train.py --input <proc_data_dir/> --logdir <logdir/> 
```