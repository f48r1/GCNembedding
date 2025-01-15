configs={
    # [ ] Set type of method hold out cv ? Stratified, Shuffle ...
    'n_split':{
        'type':int,
        'required':False,
        'default':30,
        'help':'Endpoint dataset',
    },
    'test_size':{
        'type':float,
        'required':False,
        'default':0.1,
        'help':'Size of validation test for each split',
    },
    'seed':{
        'type':int,
        'required':False,
        'default':0,
        'help':'Seed for split and repetitions',
    },
}