configs = {
    'epochs':{
        'type':int,
        'default':1000,
        'required':False,
        'help':'epochs train model'
    },
    'embedding_size':{
        'type':int,
        'default':2,
        'required':False,
        'help':'number of final descriptors'
    },
    'hidden':{
        'type':int,
        'default':10,
        'required':False,
        'help':'hidden nodes for layer'
    },
    'lr':{
        'type':float,
        'default':0.001,
        'required':False,
        'help':'learning rate'
    },
}