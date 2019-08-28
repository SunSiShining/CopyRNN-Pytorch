import os
import sys
import time

import json
import torch
import logging
import argparse


# ------------------------------------------------------------------------------------------------------------
# options for logging setting
# ------------------------------------------------------------------------------------------------------------

def init_logging(logger_name, log_file, redirect_to_stdout=False, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )

    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(level)

    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.setLevel(level)

    if redirect_to_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)

    logger.info('Initializing logger: %s' % logger_name)
    logger.info('Making log output file: %s' % log_file)
    logger.info(log_file[: log_file.rfind(os.sep)])

    return logger



# ------------------------------------------------------------------------------------------------------------
# options for training initilization
# ------------------------------------------------------------------------------------------------------------
def init_opt(description, opt):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # set several options
    # ---------------------
    preprocess_opts(parser)
    model_opts(parser)    
    train_opts(parser)
    predict_opts(parser)
    # ----------------------
    
    opt = parser.parse_args(opt)
#     opt = parser.parse_args()

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    # setting in train_opts
    # Train with Maximum Likelihood or not
    if hasattr(opt, 'train_ml') and opt.train_ml:
        opt.exp += '.ml'

    # Train with Reinforcement Learning or not
    if hasattr(opt, 'train_rl') and opt.train_rl:
        opt.exp += '.rl'

    # whether use copy attention
    if hasattr(opt, 'copy_attention') and opt.copy_attention:
        opt.exp += '.copy'

    # fill time into the name: opt.exp_path = exp/%s.%s
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)

    # Path to outputs of predictions.
    setattr(opt, 'pred_path', os.path.join(opt.exp_path, 'pred/'))
    # Path to checkpoints.
    setattr(opt, 'model_path', os.path.join(opt.exp_path, 'model/'))
    # Path to log output.
    setattr(opt, 'log_path', os.path.join(opt.exp_path, 'log/'))
    setattr(opt, 'log_file', os.path.join(opt.log_path, 'output.log'))
    # Path to plots.
    setattr(opt, 'plot_path', os.path.join(opt.exp_path, 'plot/'))

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    if not os.path.exists(opt.plot_path):
        os.makedirs(opt.plot_path)

    if opt.exp.startswith('kp20k'):
#         opt.test_dataset_names = ['inspec', 'nus', 'semeval', 'krapivin', 'kp20k', 'duc']
        opt.test_dataset_names = ['kp20k']
    elif opt.exp.startswith('stackexchange'):
        opt.test_dataset_names = ['stackexchange']
    elif opt.exp.startswith('twacg'):
        opt.test_dataset_names = ['twacg']
    else:
        raise Exception('Unsupported training data')

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        train_from_model_dir = opt.train_from[:opt.train_from.rfind('model/') + 6]
        prev_opt = torch.load(
            open(os.path.join(train_from_model_dir, opt.exp + '.initial.config'), 'rb')
        )
        prev_opt.seed = opt.seed
        prev_opt.train_from = opt.train_from
        prev_opt.save_model_every = opt.save_model_every
        prev_opt.run_valid_every = opt.run_valid_every
        prev_opt.report_every = opt.report_every
        prev_opt.test_dataset_names = opt.test_dataset_names

        prev_opt.exp = opt.exp
        prev_opt.vocab_path = opt.vocab_path
        prev_opt.exp_path = opt.exp_path
        prev_opt.pred_path = opt.pred_path
        prev_opt.model_path = opt.model_path
        prev_opt.log_path = opt.log_path
        prev_opt.log_file = opt.log_file
        prev_opt.plot_path = opt.plot_path

        for k,v in vars(opt).items():
            if not hasattr(prev_opt, k):
                setattr(prev_opt, k, v)

        opt = prev_opt
    else:
        torch.save(opt,
                   open(os.path.join(opt.model_path, opt.exp + '.initial.config'), 'wb')
                   )
        json.dump(vars(opt), open(os.path.join(opt.model_path, opt.exp + '.initial.json'), 'w'))

    return opt





# ------------------------------------------------------------------------------------------------------------
# options for preprocess processing
# ------------------------------------------------------------------------------------------------------------
def preprocess_opts(parser):
    
    # source dir
    parser.add_argument('-dataset_dir', type=str, default='/data3/private/sunsi/keyphrase',
                        help="The path to the source data (raw json).") 
    parser.add_argument('-dataset_name', type=str, default='kp20k',
                        help="Name of dataset")
    
    # Dictionary Options
    parser.add_argument('-vocab_size', type=int, default=50000,
                        help="Size of the source vocabulary")
    # for copy model
    parser.add_argument('-max_unk_words', type=int, default=1000,
                        help="Maximum number of unknown words the model supports (mainly for masking in loss).")

    parser.add_argument('-words_min_frequency', type=int, default=0)

    # Length filter options
    parser.add_argument('-max_src_seq_length', type=int, default=300,
                        help="Maximum source sequence length")
    parser.add_argument('-min_src_seq_length', type=int, default=20,
                        help="Minimum source sequence length")
    parser.add_argument('-max_trg_seq_length', type=int, default=6,
                        help="Maximum target sequence length to keep.")
    parser.add_argument('-min_trg_seq_length', type=int, default=None,
                        help="Minimun target sequence length to keep.")

    # Truncation options
    parser.add_argument('-src_seq_length_trunc', type=int, default=None,
                        help="Truncate source sequence length.")
    parser.add_argument('-trg_seq_length_trunc', type=int, default=None,
                        help="Truncate target sequence length.")
    parser.add_argument('-trg_num_trunc', type=int, default=4,
                        help="Truncate examples with many targets to maximize the utility of GPU memory.")

    # Data processing options
    parser.add_argument('-shuffle', type=int, default=1,
                        help="Shuffle data")
    parser.add_argument('-lower', default=True,
                        action = 'store_true', help='lowercase data')

    # Options most relevant to summarization
    parser.add_argument('-dynamic_dict', default=True,
                        action='store_true', help="Create dynamic dictionaries (for copy)")
    
    
    
    
# ------------------------------------------------------------------------------------------------------------
# options for model parameters
# ------------------------------------------------------------------------------------------------------------

def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=150,
                        help='Word embedding for both.')

    parser.add_argument('-position_encoding', action='store_true', default=True, 
                        help='Use a sin to mark relative words positions.')
    parser.add_argument('-share_decoder_embeddings', action='store_true',
                        help='Share the word and out embeddings for decoder.')
    parser.add_argument('-share_embeddings', action='store_true',
                        help="""Share the word embeddings between encoder
                         and decoder.""")

    # RNN Options
    parser.add_argument('-encoder_type', type=str, default='rnn',
                        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn', 'transformer', 'cnn'],
                        help='Type of decoder layer to use.')

    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=300,
                        help='Size of LSTM hidden states')
    # parser.add_argument('-input_feed', type=int, default=1,
    #                     help="""Feed the context vector at each time step as
    #                     additional input (via concatenation with the word
    #                     embeddings) to the decoder.""")

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    parser.add_argument('-input_feeding', action="store_true",
                        help="Apply input feeding or not. Feed the updated hidden vector (after attention)"
                             "as new hidden vector to the decoder (Luong et al. 2015). "
                             "Feed the context vector at each time step  after normal attention"
                             "as additional input (via concatenation with the word"
                             "embeddings) to the decoder.")

#     parser.add_argument('-bidirectional',
#                         action = "store_true",
#                         help="whether the encoder is bidirectional")
    
    parser.add_argument('-bidirectional', action = "store_true", default=True,
                        help="whether the encoder is bidirectional")    

    # Attention options
    parser.add_argument('-attention_mode', type=str, default='general',
                        choices=['dot', 'general', 'concat'],
                        help="""The attention type to use:
                        dot or general (Luong) or concat (Bahdanau)""")

    parser.add_argument('-target_attention_mode', type=str, default='general',
                        choices=['dot', 'general', 'concat', None],
                        help="""The attention type to use: dot or general (Luong) or concat (Bahdanau)""")

    # Genenerator and loss options.
    parser.add_argument('-copy_attention', action="store_true", default=True,
                        help='Train a copy model.')

    parser.add_argument('-copy_mode', type=str, default='general',
                        choices=['dot', 'general', 'concat'],
                        help="""The attention type to use: dot or general (Luong) or concat (Bahdanau)""")

    parser.add_argument('-copy_input_feeding', action="store_true",
                        help="Feed the context vector at each time step after copy attention"
                             "as additional input (via concatenation with the word"
                             "embeddings) to the decoder.")

    parser.add_argument('-reuse_copy_attn', action="store_true",
                       help="Reuse standard attention for copy (see See et al.)")

    parser.add_argument('-copy_gate', action="store_true",
                        help="A gate controling the flow from generative model and copy model (see See et al.)")

    # Cascading model options
    parser.add_argument('-cascading_model', action="store_true",
                        help='Train a copy model.')
    
    
    
    
# ------------------------------------------------------------------------------------------------------------
# options for training options
# ------------------------------------------------------------------------------------------------------------

def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-data_path_prefix', type=str, default='',
                        help="""Path prefix to the ".train.pt" and
                        ".valid.pt" file path from preprocess.py""")
    parser.add_argument('-vocab_path', type=str, default='',
                        help="""Path prefix to the ".vocab.pt"
                        file path from preprocess.py""")

    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    # GPU
    parser.add_argument('-device_ids', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=9527,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    
    
    # Init options
    parser.add_argument('-epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init).
                        Use 0 to not use initialization""")

    # Pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    # Fixed word vectors
    parser.add_argument('-fix_word_vecs_enc',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")
    parser.add_argument('-fix_word_vecs_dec',
                        action='store_true',
                        help="Fix word embeddings on the encoder side.")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Maximum batch size')
    parser.add_argument('-batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad', 'adadelta', 'adam'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=2,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-truncated_decoder', type=int, default=0,
                        help="""Truncated bptt.""")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Dropout probability; applied in LSTM stacks.")

    # Learning options
    parser.add_argument('-train_ml', action="store_true", default=True,
                        help='Train with Maximum Likelihood or not')
    parser.add_argument('-train_rl', action="store_true", default=False,
                        help='Train with Reinforcement Learning or not')
    parser.add_argument('-loss_scale', type=float, default=0.5,
                        help='A scaling factor to merge the loss of ML and RL parts: L_mixed = γ * L_rl + (1 − γ) * L_ml'
                             'The γ used by Metamind is 0.9984 in "A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION"'
                             'The α used by Google is 0.017 in "Google Translation": O_Mixed(θ) = α ∗ O_ML(θ) + O_RL(θ)'
                         )
    parser.add_argument('-rl_method', default=0, type=int,
                        help="""0: ori, 1: running average as baseline""")
    parser.add_argument('-rl_start_epoch', default=2, type=int,
                        help="""from which epoch rl training starts""")
    # GPU

    # Teacher Forcing and Scheduled Sampling
    parser.add_argument('-must_teacher_forcing', action="store_true", default=True, 
                        help="Apply must_teacher_forcing or not")
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0,
                        help="The ratio to apply teaching forcing ratio (default 0)")
    parser.add_argument('-scheduled_sampling', action="store_true",
                        help="Apply scheduled sampling or not")
    parser.add_argument('-scheduled_sampling_batches', type=int, default=10000,
                        help="The maximum number of batches to apply scheduled sampling")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_rl', type=float, default=0.0001,
                        help="""Starting learning rate for Reinforcement Learning.
                        Recommended settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=2,
                        help="""Start checkpointing every epoch after and including
                        this epoch""")
    parser.add_argument('-decay_method', type=str, default="",
                        choices=['noam'], help="Use a custom decay rate.")
    parser.add_argument('-warmup_steps', type=int, default=4000,
                        help="""Number of warmup steps for custom decay.""")

    parser.add_argument('-run_valid_every', type=int, default=2000,
                        help="Run validation test at this interval (every run_valid_every batches)")
    parser.add_argument('-early_stop_tolerance', type=int, default=10,
                        help="Stop training if it doesn't improve any more for serveral rounds of validation")

    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    parser.add_argument('-timemark', type=str, default=timemark,
                        help="Save checkpoint at this interval.")

    # output setting
    parser.add_argument('-save_model_every', type=int, default=2000,
                        help="Save checkpoint at this interval.")

    parser.add_argument('-report_every', type=int, default=100,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="kp20k",
                        help="Name of the experiment for logging.")
    parser.add_argument('-exp_path', type=str, default="./log/%s.%s",
                        help="Path of experiment log/plot.") 

    # beam search setting
    parser.add_argument('-beam_search_batch_example', type=int, default=8,
                        help='Maximum of examples for one batch, should be disabled for training')
    parser.add_argument('-beam_search_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-beam_search_batch_workers', type=int, default=4,
                        help='Number of workers for generating batches')

    parser.add_argument('-beam_size',  type=int, default=32,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=5,
                        help='Maximum sentence length.')
    
    
    
# ------------------------------------------------------------------------------------------------------------
# options for prediction options
# ------------------------------------------------------------------------------------------------------------
def predict_opts(parser):
    parser.add_argument('-must_appear_in_src', action='store_true', default=True,
                        help='whether the predicted sequences must appear in the source text')

    parser.add_argument('-report_score_names', type=str, nargs='+',
                        # default=['f_score@5_exact', 'f_score@10_exact', 'f_score@5_soft', 'f_score@10_soft'],
                        default=['f_score@5_exact', 'f_score@10_exact'],
                        help="""Default measure to report""")

    parser.add_argument('-test_dataset_root_path', type=str, default="data/")

    parser.add_argument('-test_dataset_names', type=str, nargs='+',
                        default=[],
                        help='(Set later) Name of each test dataset, also the name of folder from which we load processed test dataset.')
