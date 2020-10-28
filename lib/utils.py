import numpy as np
import os.path as osp
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
# from keras.models import load_model
import keras.optimizers
from keras.engine.network import Network
# noinspection PyPep8Naming
from keras import backend as K
from glob import glob
from collections import defaultdict
import resampy
import wav_manip as wm

# Supported window functions
WINFUNC = {None: None,
           'hamming': np.hamming,
           'hanning': np.hanning,
           'blackman': np.blackman,
           'bartlett': np.bartlett}

def proba2classes(yproba):
    ycls = np.zeros(yproba.shape, dtype='int8')
    ycls[yproba > 0.5] = 1
    return ycls


def get_scoring_func(scr_str):
    if scr_str == 'f1':
        return f1_score
    elif scr_str == 'precision':
        return precision_score
    elif scr_str == 'recall':
        return recall_score
    elif scr_str == 'roc_auc':
        return roc_auc_score
    elif scr_str == 'balanced_accuracy':
        return balanced_accuracy_score
    elif scr_str == 'average_precision':
        return average_precision_score
    else:
        raise ValueError("Scoring function for '"+scr_str+"' is not defined")


def get_best_epoch_idx(history, metric='val_loss'):
    if 'loss' in metric:
        return np.argmin(history.history[metric])
    else:
        return np.argmax(history.history[metric])


def match_metrics_and_scores(metrics_names, scores):
    res = {}
    for m, s in zip(metrics_names, scores):
        res[m] = s
    return res

# Load data
def load_data(uttlist_fname, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=None):
    # Check window function (None means rectangular window)
    try:
        winfunc = WINFUNC[winfunc]
    except KeyError:
        raise KeyError('Window fuction {} is not supported'.format(winfunc))
    # Read list of utterances to load
    with open(uttlist_fname, 'rt') as fr:
        uttlist = fr.read().splitlines()
    # noinspection PyPep8Naming
    X, y, _ = data_matrices_from_uttlist(uttlist, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=winfunc)
    # noinspection PyPep8Naming
    X = data_as_timesteps(X)
    shape = X.shape[1], X.shape[2]
    return X, y, shape


# Sample-based frame-length features
def data_matrices_from_uttlist(utts, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=None):
    data_list, tgt_list = [], []
    sf = None
    for un in utts:
        # noinspection PyPep8Naming
        # Read samples from an utterance
        spc, sf = wm.read(osp.join(spc_dir, un + '.wav'), 'float')
        fsize = int(frame_length * sf)
        data = wm.frame(spc, np.load(osp.join(peak_dir, un + '.npy')), fsize, winfunc)
        # Collect data
        data_list.append(data)
        tgt_list.append(np.load(osp.join(tgt_dir, un + '.npy')))
    # Make matrices
    y = np.hstack(tgt_list)
    # noinspection PyPep8Naming
    X_data = np.vstack(data_list)
    return X_data, y, sf


# noinspection PyPep8Naming
def data_as_timesteps(X):
    """Represent data as "timesteps" --> it has the shape (n_samples, timesteps, 1)

    Args:
        X ():

    Returns:
    """
    if isinstance(X, list):
        # Make tabular data from a list of examples
        X = np.vstack(X)
    # Stack data so that features are the 3rd dimension
    return np.dstack([X])


# Define & create a model
def create_model(model, loss, optimizer, metrics, summary=None):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # Write model summary to a file by passing the file handle in as a lambda function to make it callable
    if summary:
        with open(summary, 'wt') as fw:
            model.summary(print_fn=lambda x: fw.write(x + '\n'))
        # Plot model
        # plot_model(model, show_shapes=True, to_file='model_structure.png')
    return model


# Fit model
# noinspection PyPep8Naming
def fit_model(model, X_trn, y_trn, X_val, y_val, cfg):
    # Extract settings
    batch_size = cfg['train']['batch_size']
    n_epochs = cfg['train']['n_epochs']     # No. of training epochs
    early_stop_patience = cfg['train']['early_stop_patience']   # Early stop patience
    best_weight_fn = cfg['output']['best_weights_file']
    verbose = cfg['output']['verbose']
    csv_logger = cfg['output']['csv_logger']
    tb_logdir = cfg['output']['tb_logdir']
    # Init callbacks with saving the best model
    callbacks = [ModelCheckpoint(best_weight_fn, monitor='val_loss', verbose=verbose, save_best_only=True)]
    # Early stopping: None means no stopping => patience is the same as no. of epochs
    patience = n_epochs if early_stop_patience is None else early_stop_patience
    es = EarlyStopping(monitor='val_loss', patience=patience)
    callbacks.append(es)
    if tb_logdir is not None:
        callbacks.append(TensorBoard(log_dir=tb_logdir, write_graph=True))
    if csv_logger is not None:
        callbacks.append(CSVLogger(csv_logger, separator=';', append=True))
    # Fit the model
    history = model.fit(X_trn, y_trn, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        verbose=verbose, callbacks=callbacks)

    # Analyze early stopping:
    # stopped epoch 0 means that no stopping was applied => all n_epochs were used
    stopped_epoch = es.stopped_epoch if es.stopped_epoch > 0 else n_epochs
    best_epoch_idx = get_best_epoch_idx(history, 'val_loss')

    return {'history': history,
            'stopped_epoch': stopped_epoch,
            'best_epoch_idx': best_epoch_idx}


# Finalize the model
# noinspection PyPep8Naming
def finalize_model(model, X, y, cfg):
    # Fit the final model
    history = model.fit(X, y, epochs=cfg['finalize']['n_epochs'], batch_size=cfg['train']['batch_size'],
                        verbose=cfg['output']['verbose'])
    return model, history.history


# noinspection PyPep8Naming
def evaluate_model(model, X, y, cfg, dataset='', history=None, best_epoch_idx=None):
    verbose = cfg['output']['verbose']
    extra_metrics = cfg['train']['extra_metrics']

    # If history object inputted, extract the best scores according to val loss instead of re-calculating them again
    if history:
        assert('val_loss' in history.keys()), 'val_loss must be in the history object but it is not'
        (loss, acc) = ('loss', 'accuracy') if dataset == 'train' else ('val_loss', 'val_accuracy')
        # If best_epoch_idx inputted, it could be used to index best metrics from the history object. Otherwise, find
        # the best val loss in the history object.
        # WARNING: The inputted best_epoch must be decremented by 1 because the 1st epoch is indexed as 0!!!
        if best_epoch_idx is None:
            best_epoch_idx = get_best_epoch_idx(history, 'val_loss')
        if 'accuracy' in history.keys():
            scr = history[loss][best_epoch_idx], history[acc][best_epoch_idx]
        else:
            scr = history[loss][best_epoch_idx]
    else:
        # Re-evaluate using standard Keras metrics
        scr = model.evaluate(X, y, verbose=verbose)
    # Make scores a tuple even for a scalar case
    if not isinstance(scr, (list, tuple)):
        scr = (scr,)

    # Check model's metric names and output of evaluate()
    # !!! model.metric_names probably does not work in Tensorflow v2 / Keras v 2.4 !!!
    assert (len(scr) == len(model.metrics_names)), 'Metrics mismatched: {} expected ({}) but only {} evaluated'. \
        format(len(model.metrics_names), model.metrics_names, len(scr))

    # Match basic scores and metrics
    scores = match_metrics_and_scores(model.metrics_names, scr)
    # Calculate some other metrics
    if extra_metrics is None:
        extra_metrics = []
    # Predict to get some other metrics
    yproba = model.predict(X, verbose=verbose)[:, 0]
    yhat = utils.proba2classes(yproba)
    # Loop over extra metrics
    scr = [get_scoring_func(m)(y, yhat) for m in extra_metrics]
    # Update scores
    scores.update(match_metrics_and_scores(extra_metrics, scr))
    return scores, yhat


# Save model
def save_model(model, arch_file=None, weights_file=None):
    # Save architecture to JSON file
    if arch_file:
        with open(arch_file, 'wt') as json_file:
            json_file.write(model.to_json())
    # Save weights to hdf5 file
    if weights_file:
        model.save_weights(weights_file)


# Re-initialize weights
# Does not work for SeparableConv
# def reset_weights(model):
#     session = K.get_session()
#     for layer in model.layers:
#         if hasattr(layer, 'kernel_initializer'):
#             layer.kernel.initializer.run(session=session)


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if isinstance(layer, Network):
            reset_weights(layer)
            continue
        for v in layer.__dict__.values():
            if hasattr(v, 'initializer'):
                v.initializer.run(session=session)


# Summarize results
def summarize_results(scores, dataset='', file=sys.stdout):
    for m, s in scores.items():
        if 'loss' in m:
            print('{:5s} {:20s}: {:7.5f} (+/-{:6.5f})'.format(dataset, m, np.mean(s), np.std(s)), file=file)
        else:
            print('{:5s} {:20s}: {:7.3%} (+/-{:7.3%})'.format(dataset, m, np.mean(s), np.std(s)), file=file)


def append_scores(tot, curr):
    for m, s in curr.items():
        tot[m].append(s)
    return tot


def write_results(fn, scores, res_fit=None):
    with open(fn, 'wt') as f:
        if res_fit:
            print('Stopped epoch: {}'.format(res_fit['stopped_epoch']), file=f)
            print('Best epoch: {}'.format(res_fit['best_epoch_idx']), file=f)
            print()
        for m, s in scores.items():
            if 'loss' in m:
                print('{:20s}= {:7.5f}'.format(m, s), file=f)
            else:
                print('{:20s}= {:7.3%}'.format(m, s), file=f)


# noinspection PyPep8Naming
# Validate model
def validate_model(model, X_trn, y_trn, X_val, y_val, cfg, n_repeats=1):
    # Base best model path
    best_weights_file = cfg['output']['best_weights_file']
    path_result = osp.splitext(cfg['output']['result_file'])
    verbose = cfg['output']['verbose']
    # Init
    res_fit, ypred_val = {}, []
    # Prepare file for output summary
    fsum = open(cfg['output']['val_summary'], 'wt')
    # Input shape
    print('Train data shape:      {}'.format(X_trn.shape), file=fsum)
    print('Validation data shape: {}'.format(X_val.shape), file=fsum)
    # Init metrics
    tot_val, tot_train = defaultdict(list), defaultdict(list)

    # Loop over the number of repeats (if specified)
    for r in range(n_repeats):
        # Reset weights for multiple repeats
        reset_weights(model)
        # Fit the model
        res_fit = fit_model(model, X_trn, y_trn, X_val, y_val, cfg)
        # Load best model, otherwise model from last epoch would be evaluated
        model.load_weights(best_weights_file)
        # Evaluate on training/validation data
        train_scores, _ = evaluate_model(model, X_trn, y_trn, cfg, 'train', res_fit['history'].history,
                                         res_fit['best_epoch_idx'])
        val_scores, ypred_val = evaluate_model(model, X_val, y_val, cfg, 'val', res_fit['history'].history,
                                               res_fit['best_epoch_idx'])
        # Write results
        write_results('{}_{}.r{}{}'.format(path_result[0], 'train', r+1, path_result[1]), train_scores, res_fit)
        write_results('{}_{}.r{}{}'.format(path_result[0], 'val', r+1, path_result[1]), val_scores, res_fit)
        # Print log
        log_line = '>#{:2d} (epoch={:3d}/{:3d}): train: loss={:7.5f} acc={:7.3%} validation: loss={:7.5f} acc={:7.3%}'.\
            format(r+1, res_fit['best_epoch_idx'], res_fit['stopped_epoch'], train_scores['loss'],
                   train_scores['accuracy'], val_scores['loss'], val_scores['accuracy'])
        if verbose:
            print(log_line)
        print(log_line, file=fsum)
        # Append val scores
        tot_val = append_scores(tot_val, val_scores)
        tot_train = append_scores(tot_train, train_scores)
        # For debugging puposes: only the last repeat will survive
    # Summarize results across the repeats
    print('---', file=fsum)
    summarize_results(tot_train, 'train', file=fsum)
    summarize_results(tot_val, 'val', file=fsum)
    fsum.close()
    return res_fit['history'].history, ypred_val


# noinspection PyPep8Naming
# Test model
def test_model(model, X, y, cfg):
    return evaluate_model(model, X, y, cfg)


def preprocess(origsamples, sf_src, sf_tgt, filtcoef, norm_amp_spc=30000, norm_amp_filt=0.9):
    samples = downsample(origsamples, sf_src, sf_tgt, norm_amp_spc)
    peaks, filtsamples = detect_peaks(samples, filtcoef, norm_amp_filt)
    return samples, peaks, filtsamples


def detect_peaks(samples, filtcoef, norm_amp_filt=0.9):
    # --- Filter speech and detect peaks
    # Filter speech signal
    filtsamples = wm.filt(samples, filtcoef, zmeansource=True, norm=norm_amp_filt)
    # Convert to int16
    # filtsamples = np.rint(filtsamples).astype('int16')
    filtsamples = wm.float2int(filtsamples)
    # Detect (negative) peaks in filtered signal
    peaks = wm.detect_peaks(filtsamples, peak_polarity='-')
    return peaks, filtsamples


def downsample(origsamples, sf_src, sf_tgt, norm_amp_spc=30000):
    # Downsample signal
    assert sf_src >= sf_tgt, 'Sampling frequency for filtering ({}) is higher than source SF ({}) Hz'. \
        format(sf_tgt, sf_src)
    if sf_tgt != sf_src:
        samples = resampy.resample(origsamples, sf_src, sf_tgt)
    else:
        samples = origsamples
    # Normalize speech signal
    samples = wm.normalize(samples, norm_amp_spc)
    return samples
