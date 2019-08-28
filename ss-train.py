# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import copy
import tqdm
import torch
import logging
import numpy as np
from torch.optim import Adam

import pykp
import utils
import evaluate


from pykp.io import KeyphraseDataset
from config import init_logging, init_opt
from beam_search import SequenceGenerator
from pykp.dataloader import KeyphraseDataLoader
from utils import Progbar, plot_learning_curve_and_write_csv

from pykp.model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionCascading
from evaluate import evaluate_beam_search, get_match_result, self_redundancy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# Load dataset & vocab
# -------------------------------------------------------------------------------------------

def load_data_vocab_for_training(opt):

    logging.info('**************************** Dataset **************************** ')
    logging.info("Loading vocab & train and validate data from disk: %s" % (opt.data_path_prefix))
    word2id, id2word, vocab = torch.load(opt.vocab_path, 'rb')
    pin_memory = torch.cuda.is_available()
    
    # ------------------------------------------------------------------------------------------------------------------------
    # Training Dataset
    train_data_path = opt.data_path_prefix + '.train.one2many.pt'
    train_one2many_dataset = KeyphraseDataset(train_data_path,
                                              word2id=word2id,
                                              id2word=id2word,
                                              type='one2many',
                                              lazy_load=False)

    train_one2many_loader = KeyphraseDataLoader(dataset=train_one2many_dataset,
                                                collate_fn=train_one2many_dataset.collate_fn_one2many,
                                                num_workers=opt.batch_workers,
                                                max_batch_example=1024,
                                                max_batch_pair=opt.batch_size,
                                                pin_memory=pin_memory,
                                                shuffle=True)

    logging.info('#(train data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d, #(average examples/batch)=%.3f' 
                 % (len(train_one2many_loader.dataset), train_one2many_loader.one2one_number(), len(train_one2many_loader), train_one2many_loader.one2one_number() / len(train_one2many_loader)))


    # ------------------------------------------------------------------------------------------------------------------------
    # Validation Dataset
    valid_dataset_path = opt.data_path_prefix + '.valid.one2many.pt'
    valid_one2many_dataset = KeyphraseDataset(valid_dataset_path,
                                              word2id=word2id,
                                              id2word=id2word,
                                              type='one2many',
                                              include_original=True,
                                              lazy_load=True)
    valid_one2many_loader = KeyphraseDataLoader(dataset=valid_one2many_dataset,
                                                collate_fn=valid_one2many_dataset.collate_fn_one2many,
                                                num_workers=opt.batch_workers,
                                                max_batch_example=opt.beam_search_batch_example,
                                                max_batch_pair=opt.beam_search_batch_size,
                                                pin_memory=pin_memory,
                                                shuffle=False)

    logging.info('#(valid data size: #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' 
                 % (len(valid_one2many_loader.dataset), valid_one2many_loader.one2one_number(), len(valid_one2many_loader)))

    
        
    # ------------------------------------------------------------------------------------------------------------------------
    # Testing Dataset
    test_dataset_path = opt.data_path_prefix + '.test.one2many.pt'
    test_one2many_dataset = KeyphraseDataset(test_dataset_path,
                                             word2id=word2id,
                                             id2word=id2word,
                                             type='one2many',
                                             include_original=True,
                                             lazy_load=True)
    test_one2many_loader = KeyphraseDataLoader(dataset=test_one2many_dataset,
                                               collate_fn=test_one2many_dataset.collate_fn_one2many,
                                               num_workers=opt.batch_workers,
                                               max_batch_example=opt.beam_search_batch_example,
                                               max_batch_pair=opt.beam_search_batch_size,
                                               pin_memory=pin_memory,
                                               shuffle=False)
    logging.info('#(test data size:  #(one2many pair)=%d, #(one2one pair)=%d, #(batch)=%d' 
                 % (len(test_one2many_loader.dataset), test_one2many_loader.one2one_number(), len(test_one2many_loader)))

    # ------------------------------------------------------------------------------------------------------------------------
    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    logging.info('#(vocab from data)=%d' % len(vocab))
    logging.info('#(vocab in setting)=%d' % opt.vocab_size)
    
    if opt.vocab_size > len(vocab):
        logging.info('size of vocab is smaller than setting, reset it to %d' % len(vocab))
        opt.vocab_size = len(vocab)
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return train_one2many_loader, valid_one2many_loader, test_one2many_loader




# -------------------------------------------------------------------------------------------
# initilize model & optimizer
# -------------------------------------------------------------------------------------------

def init_model(opt):
    logging.info('**************************** Model **************************** ')
    if opt.cascading_model:
        model = Seq2SeqLSTMAttentionCascading(opt)
    else:
        if opt.copy_attention:
            logging.info('Train a Seq2Seq model with Copy Mechanism')
        else:
            logging.info('Train a normal Seq2Seq model')
        model = Seq2SeqLSTMAttention(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        if torch.cuda.is_available():
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        model.load_state_dict(checkpoint)
    else:
        # dump the meta-model
        torch.save(
            model.state_dict(),
            open(os.path.join(opt.train_from[: opt.train_from.find('.epoch=')], 'initial.model'), 'wb')
        )

    utils.tally_parameters(model)
    
    # ----------------------------------------------------------------------------
    # optimizer
    optimizer = Adam(params=filter(lambda p: p.requires_grad, 
                                   model.parameters()), lr=opt.learning_rate)
    criterion = torch.nn.NLLLoss(ignore_index=opt.word2id[pykp.io.PAD_WORD])
    
    # ----------------------------------------------------------------------------
    # run on gpu
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()
        logging.info('Running on GPU! devices=%s' % str(opt.gpuid))
        
#     if len(opt.gpuid) > 1:
#         model = torch.nn.DataParallel(model)

    # ----------------------------------------------------------------------------
    # Sequence Generator
    generator = SequenceGenerator(model,
                                  eos_id=opt.word2id[pykp.io.EOS_WORD],
                                  beam_size=opt.beam_size,
                                  max_sequence_length=opt.max_sent_length
                                 )

    return model, optimizer, criterion, generator


# -------------------------------------------------------------------------------------------
# training mini-process
# -------------------------------------------------------------------------------------------

def train_process(one2one_batch, model, optimizer, criterion, opt):
    
    src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists = one2one_batch
    max_oov_number = max([len(oov) for oov in oov_lists])

    if torch.cuda.is_available():
        src = src.cuda()
        trg = trg.cuda()
        trg_target = trg_target.cuda()
        trg_copy_target = trg_copy_target.cuda()
        src_oov = src_oov.cuda()

    optimizer.zero_grad()

    try:
        decoder_log_probs, _, _ = model.forward(src, src_len, trg, src_oov, oov_lists)

        # simply average losses of all the predicitons
        # IMPORTANT, must use logits instead of probs to compute the loss, otherwise it's super super slow at the beginning (grads of probs are small)!
        start_time = time.time()

        if not opt.copy_attention:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1, opt.vocab_size),
                trg_target.contiguous().view(-1)
            )
        else:
            loss = criterion(
                decoder_log_probs.contiguous().view(-1, opt.vocab_size + max_oov_number),
                trg_copy_target.contiguous().view(-1)
            )

        start_time = time.time()
        loss.backward()

        if opt.max_grad_norm > 0:
            pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            after_norm = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)

        optimizer.step()

        if torch.cuda.is_available():
            loss_value = loss.cpu().data.numpy()
        else:
            loss_value = loss.data.numpy()

    except RuntimeError as re:
        logging.exception("Encountered a RuntimeError")
        loss_value = 0.0
        decoder_log_probs = []

    return loss_value, decoder_log_probs




# -------------------------------------------------------------------------------------------
# training main function
# -------------------------------------------------------------------------------------------

def train_model(model, optimizer, criterion, train_data_loader, opt, stats):
    
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()
    
    for batch_i, batch in enumerate(train_data_loader):
        stats['total_batch'] += 1     
        
        model.train()        
        _, one2one_batch = batch
        
        # compute 
        loss, decoder_log_probs = train_process(one2one_batch, model, optimizer, criterion, opt)
        
        train_loss.update(loss.item(), decoder_log_probs.size(0))
        stats['all_loss'].append(loss)
        
        if len(decoder_log_probs) == 0:continue
        
        if batch_i % opt.report_every == 0:
            logging.info('train: Epoch = %d | iter = %d/%d | ' %
                        (stats['epoch'], batch_i, len(train_data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, stats['timer'].time()))
            train_loss.reset()
            
    logging.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
            (stats['epoch'], epoch_time.time()))
    
    
# -------------------------------------------------------------------------------------------
# evaluation mini-process
# -------------------------------------------------------------------------------------------

def evaluate_process(one2many_batch, generator, score_dict, opt, topk_range, score_names):
    
    src_list, src_len, trg_list, _, trg_copy_target_list, src_oov_map_list, oov_list, src_str_list, trg_str_list = one2many_batch

    if torch.cuda.is_available():
        src_list = src_list.cuda()
        src_oov_map_list = src_oov_map_list.cuda()

    try:
        pred_seq_list = generator.beam_search(src_list, src_len, src_oov_map_list, oov_list, opt.word2id)
    except RuntimeError as re:
        logging.exception('Encountered OOM RuntimeError, now trying to predict one by one')
        raise re

    # -------------------------------------------------------------------------------------------
    # compute pred score one by one 
    for src, src_str, trg, trg_str_seqs, trg_copy, pred_seq, oov in zip(src_list, src_str_list, trg_list, trg_str_list, trg_copy_target_list, pred_seq_list, oov_list):
        src = src.cpu().data.numpy() if torch.cuda.is_available() else src.data.numpy()

        # -------------------------------------------------------------------------------------------
        # filtering for target true keyphrase
        trg_str_is_present_flags, _ = evaluate.if_present_duplicate_phrases(src_str, trg_str_seqs) # flag = [True, False, ...] 
        if opt.must_appear_in_src and np.sum(trg_str_is_present_flags) == 0:
            logger.error('found no present targets')
            continue

        # -------------------------------------------------------------------------------------------
        # 1st filtering for pred
        pred_is_valid_flags, processed_pred_seqs, processed_pred_str_seqs, processed_pred_score = evaluate.process_predseqs(pred_seq, oov, opt.id2word, opt)

        # -------------------------------------------------------------------------------------------
        # 2th filtering for pred
        if opt.must_appear_in_src:
            # useful pred keyphrase flags
            pred_is_present_flags, _ = evaluate.if_present_duplicate_phrases(src_str, processed_pred_str_seqs)
            # Lable groundtruth keyphrase list
            filtered_trg_str_seqs = np.asarray(trg_str_seqs)[trg_str_is_present_flags]

        else:
            pred_is_present_flags = [True] * len(processed_pred_str_seqs) # all true not filtering
            filtered_trg_str_seqs = np.asarray(trg_str_seqs)[[True] * len(trg_str_seqs)]

        # -------------------------------------------------------------------------------------------
        # 1th filter && 2th filter
        valid_and_present = np.asarray(pred_is_valid_flags) * np.asarray(pred_is_present_flags)
        match_list = evaluate.get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=processed_pred_str_seqs)
        # len(match_list) = len(processed_pred_str_seqs) ; list :  0/1

        # -------------------------------------------------------------------------------------------
        processed_pred_seqs = np.asarray(processed_pred_seqs)[valid_and_present]
        filtered_processed_pred_str_seqs = np.asarray(processed_pred_str_seqs)[valid_and_present]
        filtered_processed_pred_score = np.asarray(processed_pred_score)[valid_and_present]

        # -------------------------------------------------------------------------------------------
        # 3rd filtering for pred : one-word phrases
        num_oneword_seq = -1 # don't apply filter
        filtered_pred_seq, filtered_pred_str_seqs, filtered_pred_score = evaluate.post_process_predseqs(
            (processed_pred_seqs, filtered_processed_pred_str_seqs, filtered_processed_pred_score), num_oneword_seq)

        # -------------------------------------------------------------------------------------------
        match_list_exact = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='exact')
        match_list_soft = get_match_result(true_seqs=filtered_trg_str_seqs, pred_seqs=filtered_pred_str_seqs, type='partial')

        assert len(filtered_pred_seq) == len(filtered_pred_str_seqs) == len(filtered_pred_score) == len(match_list_exact) == len(match_list_soft)
        # match_list_exact : list : 0 / 1 : length < match_list 
        # match_list_soft : list : o.5 , 1 , 0 ...

        # -------------------------------------------------------------------------------------------
        # EVAL 1 EXACT : Precision & Recall & F1
        for topk in topk_range:
            results_exact = evaluate.evaluate(match_list_exact, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
            # results_exact is a tuples :  Pecision, Recall, F1
            for k, v in zip(score_names, results_exact):
                score_dict['%s@%d_exact' % (k, topk)].append(v)

        # -------------------------------------------------------------------------------------------
        # EVAL 2 SOFT : Precision & Recall & F1
        for topk in topk_range:
            results_soft = evaluate.evaluate(match_list_soft, filtered_pred_str_seqs, filtered_trg_str_seqs, topk=topk)
            for k, v in zip(score_names, results_soft):
                score_dict['%s@%d_soft' % (k, topk)].append(v)
                
        opt.eval_example_idx += 1

    return score_dict


# -------------------------------------------------------------------------------------------
# evaluation main function
# -------------------------------------------------------------------------------------------

def evaluate_model(generator, data_loader, opt, stats, mode):
    
    logging.info(' ***************** Start Evaluating : Mode = %s | Epoch = %d *****************' % (mode, stats['epoch']))
    eval_time = utils.Timer()
    opt.eval_example_idx = 0
    # ---------------------------------------------------------------------
    # Initilizate Score Dict
    topk_range=[5, 10]
    score_names = ['precision', 'recall', 'f_score']
    score_dict = utils.init_results(topk_range, score_names)
    # ---------------------------------------------------------------------
    # Evaluating ...
    for i, batch in enumerate(data_loader):
        
#         if i > 5: break
        
        one2many_batch, _ = batch
        score_dict = evaluate_process(one2many_batch, generator, score_dict, opt, topk_range, score_names)
        
        sys.stdout.write('\r %s Mode Evaluating : %d of %d (%.2f%%)' 
                         %(mode, i, len(data_loader), (i / len(data_loader)) * 100))
        sys.stdout.flush()
        

    # ---------------------------------------------------------------------
    # Logging ...
    logging.info('EVALID : Epoch = %d |  Mode = %s |  Total Examples = %d | valid time = %.2f (s)'
                 % (stats['epoch'], mode, int(opt.eval_example_idx), eval_time.time()))
    
    for topk in topk_range:
        for name in score_names:
            metric = '%s@%d_exact' %(name, topk)
            logging.info(' %s = %.2f' %(metric, np.average(score_dict[metric])))
        logging.info(' ' * 60)
        
    # ---------------------------------------------------------------------
    for topk in topk_range:
        for name in score_names:
            metric = '%s@%d_soft' %(name, topk)
            logging.info(' %s = %.3f' %(metric, np.average(score_dict[metric])))
        logging.info('-' * 60)
    
    # ---------------------------------------------------------------------
    # Return : Result 
    result = {'epoch' : stats['epoch']}
    for key, value in score_dict.items():
        result[key] = np.average(value)

    return result


# -------------------------------------------------------------------------------------------
# Main function !
# -------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # setting input parameter:
    data_dir = "/data3/private/sunsi/keyphrase/kp20k/preprocess" # preprocess_small
    preprocess_opt = "-gpuid 0 -data_path_prefix %s/kp20k -vocab_path %s/kp20k.vocab.pt" % (data_dir, data_dir)
    preprocess_opt = preprocess_opt.split()
    
    # setting argparse
    opt = init_opt(description='train.py', opt = preprocess_opt)
    logging = init_logging(logger_name='train.py', log_file=opt.log_file, redirect_to_stdout=False)
    
    logging.info('Model Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]
    
    # Setting GPU
    os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(g) for g in opt.gpuid)

    # RUN on GPU OR CPU 
    if torch.cuda.is_available():
        logging.info('Running on %s! devices=%s' % ('MULTIPLE GPUs' if len(opt.gpuid) > 1 else '1 GPU', str(opt.gpuid)))
    else:
        logging.info('Running on CPU!')

    # Loading Dataset
    train_data_loader, valid_data_loader, test_data_loader = load_data_vocab_for_training(opt)
    
    # Initial Model
    model, optimizer, criterion, generator= init_model(opt)
    
    # Training & Evaluate Model
    logging.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'total_batch': -1, 'all_loss': [], 'stop_flag': False}
    
    # Best Results
    metric_name = 'f_score' if opt.must_appear_in_src else 'recall'
    best_result = utils.init_results(score_names=[metric_name], keys=['exact'], return_num = True)
    all_results = []
    
    for epoch in range(opt.start_epoch, opt.epochs):
        stats['epoch'] = epoch
        
        # Train
        train_model(model, optimizer, criterion, train_data_loader, opt, stats)
        
#     # Evaluate after all training epoch
#     result = evaluate_model(generator, valid_data_loader, opt, stats, mode='test')
        
        # Evalute
        result = evaluate_model(generator, valid_data_loader, opt, stats, mode='valid')
        all_results.append(result)
        
        for topk in [5, 10]:
            metric = '%s@%d_exact' % (metric_name, topk)
            if result[metric] > best_result[metric]:
                logging.info('Best %s = %.3f (epoch = %d)' % (metric, result[metric], stats['epoch']))
                best_result[metric] = result[metric]
                
    with open('./all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f)
    f.close()
        
        

        