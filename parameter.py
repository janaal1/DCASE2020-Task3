# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='C:\\JAVIER\\code\\DCASE2020-Task3\\base_folder',  # Base folder containing the foa/mic and metadata folders
        #dataset_dir='/content/gdrive/My Drive/DCASE2020-Task3/base_folder',

        # OUTPUT PATH
        feat_label_dir='C:\\JAVIER\\code\\DCASE2020-Task3\\input_feature\\baseline_log_mel',  # Directory to dump extracted features and labels
        #feat_label_dir='/content/gdrive/My Drive/DCASE2020-Task3/input_feature/gammatone_nomax_gcclogmel',
        model_dir='C:\\JAVIER\\code\\DCASE2020-Task3\\outputs\\ratio-1\\models',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='C:\\JAVIER\\code\\DCASE2020-Task3\\outputs\\ratio-1\\results',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='eval',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        #AUDIO REPRESENTATION TYPE (+)
        is_gammatone=False, # if set to True, extracts gammatone representation instead of Log-Mel
        fmin=.0,

        # DNN MODEL PARAMETERS
        label_sequence_length=60,        # Feature sequence length
        batch_size=64,              # Batch size
        dropout_rate=0,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        # CNN squeeze-excitation parameter (+)
        do_baseline=False,
        ratio=16,

        # Get dataset
        folder='normalized',

        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 1000.],     # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=50,               # Train for maximum epochs
        epochs_per_fit=5,           # Number of epochs per fit
        doa_objective='masked_mse',     # supports: mse, masked_mse. mse- original seld approach; masked_mse - dcase 2020 approach
        
        #METRIC PARAMETERS
        lad_doa_thresh=20
       
    )
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    params['unique_classes'] = {
            'alarm': 0,
            'baby': 1,
            'crash': 2,
            'dog': 3,
            'engine': 4,
            'female_scream': 5,
            'female_speech': 6,
            'fire': 7,
            'footsteps': 8,
            'knock': 9,
            'male_scream': 10,
            'male_speech': 11,
            'phone': 12,
            'piano': 13
        }


    # ########### User defined parameters ##############
    # if argv == '1':
    #     print("USING DEFAULT PARAMETERS\n")

    # elif argv == '2':
    #     params['mode'] = 'dev'
    #     params['dataset'] = 'mic'

    # elif argv == '3':
    #     params['mode'] = 'eval'
    #     params['dataset'] = 'mic'

    # elif argv == '4':
    #     params['mode'] = 'dev'
    #     params['dataset'] = 'foa'

    # elif argv == '5':
    #     params['mode'] = 'eval'
    #     params['dataset'] = 'foa'

    # elif argv == '999':
    #     print("QUICK TEST MODE\n")
    #     params['quick_test'] = True
    #     params['epochs_per_fit'] = 1

    # else:
    #     print('ERROR: unknown argument {}'.format(argv))
    #     exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
