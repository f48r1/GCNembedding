from ..paths import (DIR_DATA,
                     DIR_RAWDATASET,
                     DIR_MATRIX,
                     DIR_ADJACENCY,
                     DIR_COMMUNITIES,
                     DIR_INTRAIDXS,
                     DIR_EXPERIMENTS)

configs={
    'endpoint':{
        'type':str,
        'required':True,
        'help':'Endpoint dataset',
    },
    'fp':{ # FIXME nell'argparse era chiamato come 'type'
        'type':str,
        'required':False,
        'default':"daylight",
        'choices':["smarts","daylight"],
        'help':'Endpoint dataset',
    },
    'dir_data':{
        'type':str,
        'required':False,
        'default':DIR_DATA,
        'help':'directory where raw and processed data are stored',
    },
    'dir_rawdataset':{
        'type':str,
        'required':False,
        'default':DIR_RAWDATASET,
        'help':'directory where raw dataset (csv files) are stored',
    },
    'dir_matrix':{
        'type':str,
        'required':False,
        'default':DIR_MATRIX,
        'help':'directory where matrix are stored',
    },
    'dir_adjacency':{
        'type':str,
        'required':False,
        'default':DIR_ADJACENCY,
        'help':'directory for adjacency matrix output storage',
    },
    'dir_communities':{
        'type':str,
        'required':False,
        'default':DIR_COMMUNITIES,
        'help':'directory for communities analysis storage',
    },
    'dir_intraIdxs':{
        'type':str,
        'required':False,
        'default':DIR_INTRAIDXS,
        'help':'directory for molecules indexes intra domain',
    },
    'dir_experiment':{
        'type':str,
        'required':False,
        'default':DIR_EXPERIMENTS,
        'help':'directory for molecules indexes intra domain',
    },
    'save_frequency':{
        'type':int,
        'required':False,
        'default':-1,
        'help':"frequency of saving embedding per epoch",
    },
}