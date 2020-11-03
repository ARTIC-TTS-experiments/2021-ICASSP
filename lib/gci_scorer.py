# -*- coding: utf-8 -*-
import numpy as np
import logging
import os.path
import warnings
from scipy.io import wavfile
from collections import defaultdict
from pm import OnePm, Pm
from pm_compare import PM_Compare
from gci_utils import insert_trans_gci, seconds2samples, samples2seconds, sync_gci_to_samp_peak


class Scorer(object):
    """
    Object for evaluating GCI detection accuracy.

    Attributes:
        logger (logging.Logger):        Logger
        _dist_threshold (float/int):    Distance threshold (sec) for a tested GCI.
            If the tested GCI is closer than :attr:`dist_threshold` to the corresponding reference GCI, no misdetection
            is applied. If float, absolute distance in seconds is taken; otherwise (int), percentage of actual T0 is
            taken, i.e., `dist_threshold*T0`. The actual T0 is computed from reference GCIs.
        _scoring (str):         A string specifying the evaluation measure ('idr', 'far', 'mr', 'ida', 'iacc', 'acc').
        _min_t0 (float):        Minimum T0 (seconds).
        _n_refs (int):          Number of reference GCIs.
        _n_tsts (int):          Number of tested GCIs.
        _n_dels (int):          Number of deleted GCIs (GCIs occurring in tested GCIs but not in reference GCIs).
        _n_inss (int):          Number of inserted GCIs (GCIs occurring in reference GCIs but not in tested GCIs).
        _n_shfs (int):          Number of shifted GCIs (GCIs occurring in tested GCIs but with the distance to the
            nearest reference GCI out of the given limit :attr:`dist_threshold`).
        _errors (list):         list of detection errors in seconds (float)
        _utts (:obj:`Utts`):    Utterances object. Needed to evaluate detection measures utterance-wise.

    """
    logger = logging.getLogger('gci_detect.Scorer')

    def __init__(self, dist_threshold=0.00025, scoring='idr', min_t0=0.020, sync_le=0.0025, sync_ri=0.001):
        """Init method.

        Args:
            dist_threshold (float/int): Distance threshold (sec) for a tested GCI.
                If the tested GCI is closer than :attr:`dist_threshold` to the corresponding reference GCI, no
                misdetection is applied. If float, absolute distance in seconds is taken; otherwise (int), percentage of
                actual T0 is taken, i.e., `dist_threshold*T0`. The actual T0 is computed from reference GCIs.
            scoring (str):  A string specifying the evaluation measure ('idr', 'far', 'mr', 'ida', 'iacc', 'acc').
            min_t0 (float): Minimum T0 (seconds).
            sync_le (float): Time in seconds to the left for syncing a predicted GCI with a sample peak
            sync_ri (float): Time in seconds to the right for syncing a predicted GCI with a sample peak

        """
        self._scoring = scoring
        self._n_refs = 0
        self._n_tsts = 0
        self._n_dels = 0
        self._n_inss = 0
        self._n_shfs = 0
        self._errors = []
        self.clear()

        self._utts = None

        self._dist_threshold = dist_threshold
        self._min_t0 = min_t0
        self._sync_le = sync_le
        self._sync_ri = sync_ri

    def __str__(self):
        return 'Scoring function: {}'.format(self.scoring)

    def clear(self):
        """Clear the scorer.

        """
        self.logger.debug('Clearing the scorer')
        self._n_refs = 0
        self._n_tsts = 0
        self._n_dels = 0
        self._n_inss = 0
        self._n_shfs = 0
        self._errors = []

    # Value of type float is meant to express absolute distance threshold
    # Value of type int is meant to express percentual distance threshold
    @property
    def dist_threshold(self):
        """float/int: Distance threshold (seconds) for a tested GCI."""
        return self._dist_threshold

    @dist_threshold.setter
    def dist_threshold(self, value):
        if not isinstance(value, (float, int)):
            raise ValueError('Difference threshold to compare corresponding GCIs must be either int or float but is {}'.
                             format(type(value)))
        self._dist_threshold = value

    @property
    def scoring(self):
        """str: String identifying the scoring."""
        return self._scoring

    @scoring.setter
    def scoring(self, spec):
        self._scoring = spec

    @property
    def min_t0(self):
        """float: Minimum T0 (seconds)"""
        return self._min_t0

    @min_t0.setter
    def min_t0(self, value):
        if value < 0.002:
            raise ValueError('MinT0={} is probably too low'.format(value))
        elif value > 0.020:
            raise ValueError('MinT0={} is probably too high'.format(value))
        else:
            self._min_t0 = value

    # TODO: also satisfy condifition for self.scoring == 'acc'?
    def need_t0(self):
        """boolean: Whether the parameter `min_t0` is needed (it is not needed when `dist_theshold` is float. """
        return True if isinstance(self._dist_threshold, int) else False

    def compare(self, gci_refr, gci_test):
        """Compare two pitch-mark object: a tested one vs. reference one.

        Fill in auxilliary measures used for the final evaluation:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        Returns:
            :obj:`pm_compare.PM_Compare`: Pitch-mark comparison object.

        """
        warnings.warn('Method Scorer.compare() is deprecated. Use Scorer.compare_and_accumulate() instead!',
                      DeprecationWarning)
        # print('Method Scorer.compare() is deprecated. Use Scorer.compare_and_accumulate() instead!', file=sys.stderr)
        if self.need_t0():
            if not gci_refr.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to reference GCIs')
                gci_refr = insert_trans_gci(gci_refr, self._min_t0)
            if not gci_test.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to tested GCIs')
                gci_test = insert_trans_gci(gci_test, self._min_t0)
        # Init pitch-mark comparison object
        cmp = PM_Compare(diff_t0=self._dist_threshold) if self.need_t0() else PM_Compare(diff_abs=self._dist_threshold)
        # Make the comparison
        cmp.compare_pmSeq(gci_refr, gci_test)
        # Store operations
        inss = set(x[cmp.outp_refr_pm] for x in cmp.inserted({cmp.outp_refr_pm}))
        # dels = [x[cmp.outp_test_pm] for x in cmp.tested({cmp.outp_test_pm})]
        dels = cmp.deleted()
        refs = cmp.reference(({cmp.outp_refr_pm, cmp.outp_dist_pm}))
        tsts = cmp.tested(({cmp.outp_test_pm, cmp.outp_dist_pm}))
        shfs = cmp.shifted(items={cmp.outp_refr_pm, cmp.outp_test_pm})

        self._n_refs += len(refs)
        self._n_tsts += len(tsts)
        self._n_dels += len(dels)
        self._n_inss += len(inss)
        self._n_shfs += len(shfs)

        self.logger.debug('GCI comparison results: refs={}, inserts={}, deletes={}, shifts={}'.
                          format(len(refs), len(inss), len(dels), len(shfs)))
        self.logger.debug('Shifted {}:'.format(shfs))

        if self.scoring == 'ida':
            errs = [x[cmp.outp_dist_pm] for x in refs if x[cmp.outp_refr_pm] not in set(inss)]
            self.logger.debug('Shifting errors (ms): {}'.format(np.array(errs) * 1000))
            self._errors.extend(errs)
        # return the comparison object
        return cmp

    def compare_gci(self, gci_refr, gci_test):
        """Compare two pitch-mark object: a tested one vs. reference one.

        Fill in auxilliary measures used for the final evaluation:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        Returns:
            :obj:`pm_compare.PM_Compare`: Pitch-mark comparison object.

        """
        if self.need_t0():
            if not gci_refr.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to reference GCIs')
                gci_refr = insert_trans_gci(gci_refr, self._min_t0)
            if not gci_test.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to tested GCIs')
                gci_test = insert_trans_gci(gci_test, self._min_t0)
        # Init pitch-mark comparison object
        cmp = PM_Compare(diff_t0=self._dist_threshold) if self.need_t0() else PM_Compare(diff_abs=self._dist_threshold)
        # Make the comparison
        cmp.compare_pmSeq(gci_refr, gci_test)
        # Return the comparison object
        return cmp

    def accumulate_cmps(self, cmp):
        """Accumulate comparison measures from a comparison object :obj:`pm_compare.PM_Compare`:.

        The following measures are accumulated:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            cmp (:obj:``pm_compare.PM_Compare``): Pitch-mark comparison object.
        """
        # Store operations
        inss = set(x[cmp.outp_refr_pm] for x in cmp.inserted({cmp.outp_refr_pm}))
        # dels = [x[cmp.outp_test_pm] for x in cmp.tested({cmp.outp_test_pm})]
        dels = cmp.deleted()
        refs = cmp.reference(({cmp.outp_refr_pm, cmp.outp_dist_pm}))
        tsts = cmp.tested(({cmp.outp_test_pm, cmp.outp_dist_pm}))
        shfs = cmp.shifted(items={cmp.outp_refr_pm, cmp.outp_test_pm})

        # Accumulate comparison measures
        self._n_refs += len(refs)
        self._n_tsts += len(tsts)
        self._n_dels += len(dels)
        self._n_inss += len(inss)
        self._n_shfs += len(shfs)

        self.logger.debug('GCI comparison results: refs={}, inserts={}, deletes={}, shifts={}'.
                          format(len(refs), len(inss), len(dels), len(shfs)))
        self.logger.debug('Shifted {}:'.format(shfs))

        if self.scoring == 'ida':
            errs = np.array([x[cmp.outp_dist_pm] for x in refs if x[cmp.outp_refr_pm] not in set(inss)
                             and x[cmp.outp_dist_pm] > -1])
            self.logger.debug('Shifting errors (ms): {}'.format(errs[errs > 0] * 1000))
            self._errors.extend(errs)

    def compare_and_accumulate(self, gci_refr, gci_test):
        """Compare two pitch-mark objects and accumulate comparison measures.

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        """
        cmp = self.compare_gci(gci_refr, gci_test)
        self.accumulate_cmps(cmp)
        return cmp

    @property
    def n_reference(self):
        """int: Number of reference GCIs"""
        return self._n_refs

    @property
    def n_tested(self):
        """int: Number of tested GCIs"""
        return self._n_tsts

    @property
    def n_deletes(self):
        """int: Number of deletes - GCIs occurring in tested GCIs but not in reference GCIs"""
        return self._n_dels

    @property
    def n_inserts(self):
        """int: Number of inserts - GCIs occurring in reference GCIs but not in tested GCIs"""
        return self._n_inss

    @property
    def n_shifts(self):
        """int: Number of shifts - GCIs occurring in tested GCIs but they distance to the nearest reference GCI is out
        of the given limit :attr:`dist_threshold` set up in the :meth:`__init__` or :meth:`dist_threshold` methods."""
        return self._n_shfs

    @property
    def n_matched(self):
        """int: Number of matched GCIs - GCIs occuring both in tested and reference GCIs regardless to the distance
        between` them"""
        return self._n_refs - self._n_inss

    # False alarm rate measure
    def false_alarm_rate_error(self):
        """False alarm rate error (FAR).

        Returns:
            float: False alarm rate error (FAR).

        """
        return float(self._n_dels) / self._n_refs

    # Miss rate
    def miss_rate_error(self):
        """Miss rate error (MR).

        Returns:
            float: Miss rate error (MR).

        """
        return float(self._n_inss) / self._n_refs

    # Identification rate
    def identification_rate_score(self):
        """Identification rate score (IDR).

        Returns:
            float: Identification rate score (IDR).

        """
        return float(self._n_refs - self._n_dels - self._n_inss) / self._n_refs

    # Identification accuracy to +/- diff_threshold (defined by self._dist_threshold, eg. 0.25 ms)
    # The percentage of detections for which the identification error x <= diff_threshold (the timing error between the
    # detected and the corresponding reference GCI)
    def identification_accuracy_score(self):
        """Identification accuracy (IACC) to +/- distance threshold :attr:`dist_threshold` (e.g., 0.00025 s)

        The percentage of detections for which the identification error `x <= dist_threshold` (the timing error between
        the tested and the corresponding reference GCI).

        Returns:
            float: Identification accuracy score (IACC)

        """
        return 1 - float(self._n_shfs)/self.n_matched

    def identification_accuracy_error(self):
        """Identification accuracy error (IDA).

        Returns:
            float: Identification accuracy error (IDA) in seconds.
        """
        if not self._errors:
            raise RuntimeError('Cannot apply IDA error measure: errors do not exist')
        return np.array(self._errors).std()

    # Accuracy score
    def accuracy_score(self):
        """Accuracy score (ACC)

        Computed as `(n_refs - n_shfs - n_inss - n_dels) / n_refs`.

        Returns:
            float: Accuracy score (ACC)

        """
        return (self._n_refs - self._n_shfs - self._n_inss - self._n_dels)/float(self._n_refs)

    def _score(self):
        """Helper score function that returns the score as required by the :attr:`_scoring` attribute.

        Notes:
            In `scikit-learn scoring <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_ by
            convention higher numbers are better. This is OK for measures expressing scores (such as 'idr', 'iacc',
            'acc'). For "error" or "loss" functions ('mr', 'far', 'ida'), the return value should be negated.

        Returns:
            float: The score as given by the :attr:`_scoring` attribute.

        """
        if self._scoring == 'far':
            return -self.false_alarm_rate_error()
        elif self._scoring == 'mr':
            return -self.miss_rate_error()
        elif self._scoring == 'idr':
            return self.identification_rate_score()
        elif self._scoring == 'iacc':
            return self.identification_accuracy_score()
        elif self._scoring == 'ida':
            return -self.identification_accuracy_error()
        elif self._scoring == 'acc':
            return self.accuracy_score()
        else:
            raise RuntimeError('Unsupported scoring: {}'.format(self.scoring))

    @property
    def utts(self):
        """:obj:`list`: List of utterances (:obj:`Utt`)"""
        return self._utts

    @utts.setter
    def utts(self, utts):
        self._utts = utts

    # noinspection PyPep8Naming
    # noinspection PyUnusedLocal
    def score(self, estimator, X, y):
        """Scorer callable function compatible with scikit-learn scoring function.

        Make a score for testing examples `X`. The score is defined by the scorer function :attr:`_scoring`.

        Args:
            estimator (:obj:`sklearn.base.BaseEstimator`):  estimator object implementing ‘fit’
            X (array-like):                                 The data to fit. Can be for example a list, or an array.
            y (array-like):                                 The target variable to try to predict.

        Notes:
            - No targets `y` are needed since the true GCIs, which are confronted with the predcited GCIs, are stored in
                the utterances object :attr:`_utts`.

            - `X` is a fraction that corresponds to the given split (fold) of the original data examples `X` as
                given by the cross-validation object used. As the data examples are inputted from a sckikit-learn code,
                the particular fold to be used is not known, only the data is available.
                The :meth:`_indices_examples2utt` is used to find out the mapping between the data examples and
                utterances in the partiular split `X`.

            - Although the custom cross validation :obj:`KFoldUtt` returns the mapping of each example to an utterance
                (see :meth:`KFoldUtt.example2utt_mapping`), the actual split (fold) from which the testing data examples
                `X` were picked, are not known!

        Warnings:
            The scorer function must have signature ``scorer(estimator, X, y)``.

        Returns:
            float: The detection accuracy score (or error, loss).

        """
        # Reset scorer for this testing set
        self.clear()
        # Map the testing data examples X to utterances
        ex2utt = self._indices_examples2utt(X)
        # !!!
        # Convert from list of InfoArray to Numpy array to enable "multiindexing". We are loosing the information about
        # the utterance -> example mapping by this operation, so it must be called after _indices_examples2utt() had
        # been called.
        # !!!
        X = np.array(X)
        self.logger.debug('Scoring for {}'.format(estimator))
        self.logger.debug('Scoring on set with {} utts {} and {} examples'.format(len(ex2utt),
                                                                                  list(ex2utt.keys()),
                                                                                  len(X)))
        self.logger.debug('Sampling frequency used for syncing predicted GCI positions: {}'.
                          format(self._utts.samp_freq))
        # Sampling frequency should be the same for all utterances in the dataset
        sync_le = seconds2samples(self._sync_le, self._utts.samp_freq)
        sync_ri = seconds2samples(self._sync_ri, self._utts.samp_freq)
        # Go through all utterances in this testing set and accumulate comparison indicators across the utterances
        for utt_idx, ex_indices in ex2utt.items():
            utt = self._utts[utt_idx]
            self.logger.debug('Scoring on {} examples from {} ({})'.format(len(ex_indices), utt.name, utt_idx))
            # Predict and sync GCIs for the given utterance
            pred_sync = sync_gci_to_samp_peak(predict_gci(estimator, X[np.array(ex_indices)]),
                                              utt.peaks,
                                              utt.samples,
                                              sync_le,
                                              sync_ri)
            # Set the predicted and synced GCIs as a Pm object
            # gci_pred = create_gci(samples2seconds(pred_sync, utt.samp_freq))
            gci_pred = Pm(times=samples2seconds(pred_sync, utt.samp_freq))
            # Compare GCI sequences: predicted GCIs with true GCIs (represented as a Pm object from the utt object)
            self.compare_and_accumulate(utt.ref_gcis, gci_pred)
        score = self._score()
        self.logger.debug('Score ({}) = {:8.6f} on set of {} utts {} with {} examples'.
                          format(self._scoring.upper(), score, len(ex2utt), np.array(list(ex2utt.keys())), len(X)))
        return score

    # noinspection PyPep8Naming
    @staticmethod
    def _indices_examples2utt(X):
        """Make a mapping between input testing examples and the corresponding source utterances.

        The input examples have to a list of :obj:`InfoArray` since each example ha to conatain the
        :attr:`InfoArray.info` attribute that denotes from which source utterance (given by an index) the example
        comes from.

        Args:
            X (:obj:`list`): The data to map (list of :obj:`InfoArray`).

        Returns
            dict: Dictionary with list of example indices from X for each utterance index.

        """
        d = defaultdict(list)
        for ex_idx, ex in enumerate(X):
            d[ex.info].append(ex_idx)
        return d


# noinspection PyPep8Naming
def predict_gci(estimator, X):
    """
    Predict GCI placements in seconds.

    Call the :func:`sync_gci_to_samp_peak` to sync the time placements with speech signal.

    Args:
        estimator (:obj:`sklearn.base.BaseEstimator`): Sklearn-style estimator object implementing 'fit'.
        X (array-like): The data to predict on.

    Returns:
        :obj:`numpy.array`: Array of GCI time placements (int) in samples.

    """
    return estimator.predict(X)
