#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
from glob import glob
from argparse import ArgumentParser
import os.path as osp
from keras.models import model_from_json, Model
import joblib
import librosa as lr
import gci_utils as gu
import utils


###################################################################################
#   Main                                                                          #
###################################################################################
def main():
    # Command line processing
    parser = ArgumentParser(description="Detect GCI times")
    parser.add_argument('inp',
                        help='input mask of wav files [default: %(default)s]')
    parser.add_argument('out',
                        help='output path (pitch-mark file or folder) [default: %(default)s]')
    parser.add_argument('-E', '--out_ext',
                        default='pm',
                        help='output file extension [default: %(default)s]')
    parser.add_argument('-a', '--architecture',
                        required=True,
                        help='model architecture (JSON file) [default: %(default)s]')
    parser.add_argument('-W', '--weights',
                        required=True,
                        help='model weights (hdf5 file) [default: %(default)s]')
    parser.add_argument('-e', '--feat-extraction-layer',
                        default=None,
                        type=str,
                        help='name of CNN layer for feature extraction [default: %(default)s]')
    parser.add_argument('-C', '--clf',
                        default=None,
                        help='trained classifier (joblib file) [default: %(default)s]')
    parser.add_argument('-c', '--filt-coef',
                        required=True,
                        help='filter coefficients (numpy file) [default: %(default)s]')
    parser.add_argument('-T', '--sf-tgt',
                        default=8000,
                        type=int,
                        help='sampling frequency for processing waveform (sec) [default: %(default)s]')
    parser.add_argument('-n', '--norm-amp-spc',
                        default=30000,
                        type=int,
                        help='normalization amplitude on speech signal (int) [default: %(default)s]')
    parser.add_argument('-N', '--norm-amp-filt',
                        default=0.9,
                        type=float,
                        help='normalization amplitude on filtered signal [default: %(default)s]')
    parser.add_argument('-f', '--frame-length',
                        default=0.030,
                        type=float,
                        help='frame length around a peak (sec) [default: %(default)s]')
    parser.add_argument('-w', '--winfunc',
                        default=None,
                        choices=[None, 'hamming', 'hanning', 'blackman', 'bartlett'],
                        help='frame windowing [default: %(default)s]')
    parser.add_argument('-l', '--sync-left',
                        default=0.0025,
                        type=float,
                        help='time (sec) to the left to sync predicted GCI with sample peak [default: %(default)s]')
    parser.add_argument('-r', '--sync-right',
                        default=0.0010,
                        type=float,
                        help='time (sec) to the right to sync predicted GCI with sample peak [default: %(default)s]')
    parser.add_argument('-t', '--insert-trans',
                        default=False,
                        action='store_true',
                        help='insert transitional marks [default: %(default)s]')
    parser.add_argument('-L', '--loglevel',
                        default='INFO',
                        help='logging level [default: %(default)s]')
    args = parser.parse_args()
    # Check arguments
    if args.feat_extraction_layer is not None and args.clf is None:
        parser.error('Conv net is used to extract features but no other classifier is loaded')

    ###################################################################################
    #   Logging                                                                       #
    ###################################################################################
    logger = logging.getLogger('detect_gci')
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))
    logging.basicConfig(format='%(levelname)-10s %(message)s', level=loglevel, stream=sys.stderr)

    ###################################################################################
    #   Load input data                                                               #
    ###################################################################################
    # Load saved model's architecture from JSON file and create model from the architecture
    with open(args.architecture, 'rt') as json_file:
        convnet = model_from_json(json_file.read())
    # Load model's weights from hdf5 file
    convnet.load_weights(args.weights)
    # Select layer for feature extraction
    # If layer for feature extraction is specified, conv net is used to extract features for another classifier
    if args.feat_extraction_layer:
        feat_layer = convnet.layers[args.feat_extraction_layer]
        # Define feature extractor model
        convnet = Model(inputs=convnet.input, outputs=feat_layer.output)
        # Load classifier
        clf = joblib.load(args.clf)
    else:
        clf = convnet
    # Read filter coefficients
    filtcoef = np.load(args.filt_coef)

    ###################################################################################
    #   Read input files                                                              #
    ###################################################################################
    # Read input files
    buff = glob(args.inp)
    if not buff:
        logger.warning('Input mask contains no files!')

    ###################################################################################
    #   Process all files in the input mask                                           #
    ###################################################################################
    for fn in sorted(buff):
        # --- Read waveform & prepare it for GCI detection ---
        # Set up basename
        bn = osp.splitext(osp.basename(fn))[0]
        logger.debug(bn)
        # Prepare waveform for GCI detection
        samples_src, sf_src = lr.load(fn, sr=None)
        samples_tgt, peaks, filtsamples = utils.preprocess(samples_src, sf_src, args.sf_tgt, filtcoef, args.norm_amp_spc,
                                                           args.norm_amp_filt)

        logger.debug('# peaks: {}'.format(len(peaks)))
        logger.debug('Peaks: {}'.format(peaks))
        # Check window function (None means rectangular window)
        winfunc = utils.WINFUNC[args.winfunc]
        logger.debug('Windowing: {} --> {}'.format(args.winfunc, winfunc))
        # Read speech frames into a list
        data_list = utils.frames_from_utt(samples_tgt, args.sf_tgt, peaks, args.frame_length, winfunc)
        # Reshape data to input CNN
        # noinspection PyPep8Naming
        X = utils.data_as_timesteps(data_list)
        # Extract (predict) features
        if args.feat_extraction_layer:
            # noinspection PyPep8Naming
            X = convnet.predict(X)
            # Reshape back to have tabular data again
            if X.ndim == 3:
                # noinspection PyPep8Naming
                X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

        # --- Predicition ---
        # Predict GCIs => get a prediction per a peak
        y = clf.predict(X).flatten()
        logger.debug('Predictions: {}'.format(y))
        # Convert peak indices according to SOURCE sampling frequency
        peaks_src = gu.seconds2samples(gu.samples2seconds(peaks, args.sf_tgt), sf_src)
        # Get samples of the predicted GCIs synced to the nearest peak in the SOURCE signal
        pred_samps_src = gu.sync_predictions_to_samp_peak(y, peaks_src, samples_src,
                                                          gu.seconds2samples(args.sync_left, sf_src),
                                                          gu.seconds2samples(args.sync_right, sf_src))
        pred_times_src = gu.samples2seconds(pred_samps_src, sf_src)
        logger.debug('Predicted GCI times: {}'.format(pred_times_src))

        # --- Output pitch-marks

        # Create pitch-marks from GCIs
        pms = gu.create_gci(pred_times_src)
        logger.info('{:20}: # GCIs = {}'.format(bn, len(pms)))
        if args.insert_trans:
            pms = gu.insert_trans_gci(pms)
        # Set up output path for a batch processing
        if len(buff) == 1:
            # Interpret output path as a single file path
            out_path = args.out
        else:
            # Interpret output path as a directory and append a file name
            out_path = osp.join(args.out, '{}.{}'.format(bn, args.out_ext))

        # Write GCIs to a file
        pms.write_file(out_path)


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
