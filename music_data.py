#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 06:58:24 2018

@author: arnold
"""

import os
import glob
import json
import yaml
import time
import h5py
import numpy as np
import collections
import music21 as m21
# from music21 import note
import multiprocessing as mp

class MusicData():
    # distance_to_c contains for each note in an octave the distance to C
    # in semitones. The key values of the dictionary are the note names used
    # in the program. The trans_note dictionary translates deviating note values
    # to the 12 key values used in distance_to_c

    # Declaration of static variables
    distance_to_c = {'C': 0, 'C#': -1, 'D': -2, 'D#': -3, 'E': -4, 'F': -5, 'F#': -6,
                     'G': -7, 'G#': -8, 'A': -9, 'A#': -10, 'B': -11}
    ordered_dist = collections.OrderedDict(sorted(distance_to_c.items()))

    translate_note = {'CB': 'B',  'C': 'C', 'CS': 'C#', 'C#': 'C#',
                      'DB': 'C#', 'D': 'D', 'D#': 'D#', 'D-': 'C#',
                      'EB': 'D#', 'E': 'E', 'E-': 'D#',
                      'FB': 'E',  'F': 'F', 'F-': 'E', 'FS': 'F#', 'F#': 'F#',
                      'GB': 'F#', 'G': 'G', 'G#': 'G#',
                      'AB': 'G#', 'A': 'A', 'A#': 'A#', 'A-': 'G#',
                      'BB': 'A#', 'B': 'B', 'B-': 'A#'}
    translate_mode = {'minor': 'minor', 'major': 'major', 'mineur': 'minor', 'majeur': 'major'}
    voice_names = ['Soprano', 'Alt', 'Tenor', 'Bass']
    n_vocab = 128 # MIDI notes
    octave_size = 12
    correction_for_a = +9 # value to be added for transposing a note from C-major
                          # to a minor
    def __init__(self, d_unit=8, vocab=128, n_durs=128, required=4, seq_length=50):
        self.duration_unit = d_unit
        self.n_durs = n_durs
        self.minimum_length = seq_length # minimum_length
        self.required = required
        self.sequence_length = seq_length
        self.verbose = 0
        MusicData.n_vocab = vocab

        return

    @staticmethod
    def get_score(filename):
        try:
            path = os.path.basename(filename)
            #print('Reading:', os.getpid(), path)
            score = m21.converter.parse(filename)

            # Add custom attributes to score, all prefixed by mld_
            score.mld_fullpath = filename
            score.mld_filename = os.path.splitext(path)[0]
            score.mld_semi_tones = 0
            score.mld_key = None
            score.mld_staffs = 0
            print(score.mld_filename)
        except Exception as err:
            score = None
            print(os.getpid(), path, 'MidiError: ', err)

        return score

    @staticmethod
    def get_scores(dirname, sample_size, verbose=0, n_processes=0):
        """ Read all MIDI files

        Get all the notes and chords from the MIDI files in a directory.
        When a MIDI file is uncorrectly formatted (which unfortunately not
        infrequently happens) it is skipped.
        What this function also does is adding the filename of the MIDI
        for that score as an attribute to that score. That file name
        is used later on to determine the key from the file name and
        to have some identification for the analysis results for that score.

        Args:
            dirname (str): name of directory to read scores from and from all
                           its subdirectories
                           -or-
                    (list): a list of files
            sample_size (float): < 1 fraction to sample
                                == 1 samples all files
                                 > 1 # of files to sample
        Returns:
            a list of MIDI files represented as a list of music21 scores
        """

        sec = time.time()
        if isinstance(dirname, list):
            files_to_read = dirname
            if verbose > 0:
                print('Reading', len(dirname), 'files.')
        else:
            if verbose > 0:
                print('Reading', dirname)
            files = os.path.join (dirname, "**/*.mid")
            files_found = glob.glob(files, recursive=True)
            n_files = len(files_found)
            if n_files == 0:
                print('*** No files found: check your data directory')
                print('   data:', files)

            if sample_size > n_files:
                sample_size = n_files
            elif sample_size <= 1:
                sample_size = int(sample_size * n_files)
            else:
                sample_size = int(sample_size)

            if verbose > 0:
                print(n_files, 'files found, sample of', sample_size, 'will be used')
            files_to_read = np.random.choice(files_found, size=sample_size, replace=False)

        scores = []
        results = None
        #pool = mp.Pool(int(mp.cpu_count() / 1))
        # This section is intended to skip files that cause some error
        # during reading, usually when the MIDI format is not correct
        # A keyboard error stops file reading but returns normally

        try:
            if n_processes < 2:
                print('Single threaded map')
                results = list(map(MusicData.get_score, files_to_read))
            else:
                #with mp.Pool(processes = n_processes) as pool:
                print('Test for', n_processes, 'processes')
                results = list(map(MusicData.get_score, files_to_read))

            print(len(results), 'files read into memory')
            for score in results:
                if score is not None:
                    scores.append(score)
            print(len(scores), 'files after stripping None')
        except KeyboardInterrupt:
            print('\nReading of files stopped, processing continues')

        if verbose > 0:
            sec = int(time.time() - sec + 0.5)
            print(sec, 'seconds to read scores from file')

        return scores

    @staticmethod
    def get_serial_score(filename):
        score = m21.converter.parse(filename)

        # Add custom attributes to score, all prefixed by mld_
        path = os.path.basename(filename)
        score.mld_fullpath = filename
        score.mld_filename = os.path.splitext(path)[0]
        score.mld_semi_tones = 0
        score.mld_key = None
        score.mld_staffs = 0

        return score

    def get_serial_scores(self, dirname, sample_size, verbose=0, n_processes=0):
        """ Read all MIDI files

        Get all the notes and chords from the MIDI files in a directory.
        When a MIDI file is uncorrectly formatted (which unfortunately not
        infrequently happens) it is skipped.
        What this function also does is adding the filename of the MIDI
        for that score as an attribute to that score. That file name
        is used later on to determine the key from the file name and
        to have some identification for the analysis results for that score.

        Args:
            dirname (str): name of directory to read scores from and from all
                           its subdirectories
                           -or-
                    (list): a list of files
            sample_size (float): < 1 fraction to sample
                                == 1 samples all files
                                 > 1 # of files to sample
        Returns:
            a list of MIDI files represented as a list of music21 scores
        """

        sec = time.time()
        if isinstance(dirname, list):
            files_to_read = dirname
            if verbose > 0:
                print('Reading', len(dirname), 'files.')
        else:
            if verbose > 0:
                print('Reading', dirname)
            files = os.path.join (dirname, "**/*.mid")
            files_found = glob.glob(files, recursive=True)
            n_files = len(files_found)
            if n_files == 0:
                print('*** No files found: check your data directory')
                print('   data:', files)

            if sample_size > n_files:
                sample_size = n_files
            elif sample_size <= 1:
                sample_size = int(sample_size * n_files)
            else:
                sample_size = int(sample_size)

            if verbose > 0:
                print(n_files, 'files found, sample of', sample_size, 'will be used')
            files_to_read = np.random.choice(files_found, size=sample_size, replace=False)

        i = 0
        scores = []
        for file in files_to_read:
            # This section is intended to skip files that cause some error
            # during reading, usually when the MIDI format is not correct
            # A keyboard error stops file reading but returns normally
            try:
                i += 1
                path = os.path.basename(file)
                filename = os.path.splitext(path)[0]
                if verbose > 0:
                    print("Reading {:d}: {:s}... ".format(i, filename), end = '')
                score = MusicData.get_serial_score(file)
                scores.append(score)
                if verbose > 0:
                    print()
            except KeyboardInterrupt:
                print('\nReading of files stopped, processing continues')
                break
            except Exception as err:
                print(' MidiError: ', err)
        # for

        results = []
        for score in scores:
            if score is not None:
                results.append(score)

        print(len(scores), 'files succesfully read into memory')
        sec = int(time.time() - sec + 0.5)
        if verbose > 0:
            print(sec, 'seconds to read scores from file')

        return results

    @staticmethod
    def derive_key_from_filename(filename):
        """ Returns the key signature based on file name.

        The convention used
        is that key and mode are part of the file name and are separated from
        file_name and each other by '_'. It is assumed that the last
        part of the filename is its mode and the first part the key.
        Key and mode are translated.
        examples: long_file_name_g_minor = ok
                  other_e_major_file_name = not ok, key not at the end
                  file_name-A = not ok, 'major' or 'minor' must be present

        Args:
            filename (str):file name to be analysed

        Returns:
            uppercased key and mode. None is returned when something
            is wrong.
        """
        signature = None
        parts = filename.split('_')
        if len(parts) > 2:
            try:
                modus = parts[-1].lower().split('.')[0]
                k = parts[-2].upper()
                modus = MusicData.translate_mode[modus]
                k = MusicData.translate_note[k]
                signature = m21.key.Key(k, modus)
            except:
                signature = None

        return signature

    def count_notes(self, score):
        part_count = 0
        for part in score.parts:
            n = len(part.getElementsByClass(m21.note.Note))
            n += len(part.getElementsByClass(m21.chord.Chord))

            if n > self.minimum_length:
                part_count += 1

        return part_count

    def evaluate_scores(self, scores, correction_for_minor_scale=False):
        """ Sets the key of each score

        Sets the key of each score and count the number of parts
        containing notes equaling or more than the minimum length.
        Is a key is provided by file name, this
        takes precedence over the key from the analysis from music21.

        Args:
            scores (list): list of scores to set the key for each score
            correction_for_minor_scale (bool): when true scores in minor
                key are transposed to a-minor, when false to c-minor
                The correction consists of adding 9 semitones
        Returns:
            nothing, key info is added to the score as mld_ attributes
        """
        for score in scores:
            score.mld_staffs = self.count_notes(score)
            #key_from_file, key_from_analysis = self.get_key_signature(score)
            key_from_file = self.derive_key_from_filename(score.mld_filename)
            key_from_analysis = score.analyze('key')

            if key_from_file is not None:
                score.mld_key = key_from_file
            else:
                score.mld_key = key_from_analysis

            tonica = self.translate_note[score.mld_key.tonic.name]
            score.mld_key.tonic.name = tonica
            #steps = self.distance_to_c[tonica]
            #if correction_for_minor_scale:
            #    steps += 9

        return

    def select_scores(self, all_scores, mode, voices):
        """ Selects scores

        Selects scores which mode is in mode and voices is in voices

        Args:
            all_scores(list): list of scores to be selected
            mode(list): list containing the modes to select, ['*'] is all
            voices(list): list containing the voices to select, ['*'] is all

        Returns:
            List of scores satisfying the conditions
        """
        voiced_score_list = None
        if all_scores is not None:
            print('Selecting from', len(all_scores), 'scores')
            selected_scores = []
            for score in all_scores:
                if mode == '*' or score.mld_key.mode in mode:
                    selected_scores.append(score)

            print(len(selected_scores), 'scores with scale', mode)

            # Select the scores that have the selected number of voices
            # Voices of [0] means select all scores
            voiced_score_list = []
            for score in selected_scores:
                if voices == ['*'] or score.mld_staffs in voices:
                    voiced_score_list.append(score)

            print(len(voiced_score_list), 'scores left with', str(voices), 'voices')

        return voiced_score_list

    def sample_scores(self, data_dir, sample_size, modus, voices):
        """ Selects exact number of scores satisfying the selection criteria

        This routine reads all files from the data directory and converts these
        to scores. Next selects the scores satisfying the modus and voices
        conditions.The selection is performed upon the *selected* scores.
        sample_size == 1, selects all scores
        sample_size < 1, selects fraction of the scores
        sample_size > 1, selects int(sample_size) scores

        Args:
            data_dir (str): name of data directory containg MIDI files to be
                            converted to scores
            sample_size(float or int): float - fraction of scores to be selected
                                       int - # of scores to be selected
            modus(list): contains the modi to be selected: 'major' or 'minor' or ['*']
            voices(list): number of voices a score must contain: list of integers
                          or ['*']

        Returns:
            List of scores satisfying the conditions
        """

        all_scores = self.get_serial_scores(data_dir, 1, verbose=1, n_processes=32)
        self.evaluate_scores(all_scores)
        score_list = self.select_scores(all_scores, modus, voices)

        n_files = len(score_list)
        if sample_size >= n_files:
            return score_list

        if sample_size <= 1:
            sample_size = int(sample_size * n_files)
        else:
            sample_size = int(sample_size)

        scores = np.random.choice(score_list, size=sample_size, replace=False)

        return scores

    def compute_maxlen(self, score):
        """
        Computes the maximum duration of a score in 1/64 steps.
        """
        l = len(score.parts)
        means = np.zeros((l), dtype=np.float32)
        lengths = np.zeros((l), dtype=np.int32)
        for idx, part in enumerate(score.parts):
            try: # file has instrument parts
                notes_to_parse = part.recurse()
            except: # file has notes in a flat structure
                notes_to_parse = part.flat.notes

            if len(notes_to_parse) > self.minimum_length:
                pitches = [n.pitch.midi for n in notes_to_parse if isinstance(n, m21.note.Note)]
                part_length = self.duration_unit * part.duration.quarterLength
                lengths[idx] = int(part_length)
                means[idx] = np.mean(pitches)

        return int(max(lengths))

    def get_notes_from_part(self, part, semi_tones):
        notes = []
        durations = []
        notes_to_parse = None
        n_chords = 0

        try: # file has instrument parts
            notes_to_parse = part.recurse()
        except: # file has notes in a flat structure
            notes_to_parse = part.flat.notes

        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                duration = int(self.duration_unit * element.duration.quarterLength + 0.5)
                pitch = element.pitch.midi + semi_tones
                notes.append(pitch)
                durations.append(duration)
            elif isinstance(element, m21.chord.Chord):
                duration = int(self.duration_unit * element.duration.quarterLength + 0.5)
                n = m21.note.Note(element.root())
                pitch = n.pitch.midi + semi_tones
                notes.append(pitch)
                durations.append(duration)
                n_chords += 1

        return notes, durations, n_chords

    def get_notes_from_score(self, score, minlen, transpose, correction_minor):
        """ Get notes, durations, # chords for each part of a score

        Returns for each part a list of notes, durations and number of chords,
        represented as a list of list of notes and a list of chord counts.
        Args:
            score (class): score of which the parts are to be returned
            minlen (int): minimum # of notes required for each part
            transpose (bool): True, notes are transposed
            correction_minor (bool): True, after transposing transposes down to
                                     a-minor

        Returns:
            A list with for each part a list of notes, durations and the number
            of chords counted
        """
        if transpose:
            tonica = score.mld_key.tonic.name
            steps = MusicData.distance_to_c[tonica]
            if correction_minor:
                steps += 9
        else:
            steps = 0 # same as no transposing

        parts = []
        for part in score.parts:
            # determine number of semitones to transpose
            n, d, n_chords = self.get_notes_from_part(part, steps)
            if len(n) >= self.minimum_length:
                parts.append([n, d, n_chords])

        return parts

    def get_time_series_from_part(self, part, maxlen, semi_tones):
        """ Iterate all notes and chords of a part (staff) of a score.

        There are maxlen micro time steps (usually 1/64th) and for each
        time is computed whether for that part is note is present or not.

        Args:
            part (music21 object): part (= staff) of a score
            maxlen (int): maximum length of a part in time steps

        Returns: Two values:
                 The first is a numpy array of maxlen elements where each
                 element contains the midi value for that element or zero
                 when no note was present duing that time
                 The second value is the mean of all pitches
        """
        notes = []
        notes_to_parse = None

        part_length = int(self.duration_unit * part.duration.quarterLength)
        quantized_part = part.quantize((4,3))

        timeline = np.zeros((maxlen), dtype=np.int32)

        try: # file has instrument parts
            notes_to_parse = quantized_part.recurse()
            if self.verbose > 0:
                print('\nInstrument parts', len(quantized_part), 'elements')
        except: # file has notes in a flat structure
            if self.verbose > 0:
                print('\nFlat structure', len(quantized_part), 'elements')
            notes_to_parse = quantized_part.flat.notes

        if self.verbose > 0:
            print('   duration of part is:', part.duration.quarterLength, '=', part_length, '/', maxlen)

        offset = 0
        count = 0
        som = 0
        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                timeline_marker = int(self.duration_unit * element.duration.quarterLength + 0.5)
                pitch = element.pitch.midi + semi_tones
                timeline[offset:offset+timeline_marker] = pitch
                pair = (offset, pitch)
                notes.append(pair)
                if self.verbose > 1:
                     print('Note', pair)
                offset += timeline_marker
                count += 1
                som += pitch
            elif isinstance(element, m21.note.Rest):
                timeline_marker = int(self.duration_unit * element.duration.quarterLength + 0.5)
                if self.verbose > 1:
                    print('Rest', offset)
                offset += timeline_marker
            elif isinstance(element, m21.chord.Chord):
                timeline_marker = int(self.duration_unit * element.duration.quarterLength + 0.5)
                n = m21.note.Note(element.root())
                pitch = n.pitch.midi + semi_tones
                timeline[offset:offset+timeline_marker] = pitch
                pair = (offset, pitch)
                notes.append(pair)
                if self.verbose > 1:
                    print('Chord', pair)
                offset += timeline_marker
                count += 1
                som += pitch

        mean = 0.0 if count <= 1 else som / count

        return timeline, mean

    def reorder_parts(self, parts, means):
        """
        Receives a list of all parts from a score containing self.minimum_length
        notes or more, and a list of means. The parts are reordered
        in SATB order based on means. Highest mean is
        soprano, lowest is bass, and alto and tenor in between. When there are
        no four parts the order is as follows (0 meaning an array with only
        zeros):
            1: s000
            2: s00b
            3: sa0b
            4: satb
            5: satb - sat are the three highest, b is the lowest

        Args:
            parts (list): list of parts where each part is delivered by
                 get_time_series_from_part. Meaning is is is a numpy array
                 of maxlen elements where each element contains the midi
                 value for that element or zero when no note was present
                 during that time
            means (list): mean pitch of the corresponding part

        Returns:
            a list of four parts in SATB order, each part is an array as
            delivered by get_time_series_from_part, optionally filled with
            zero arrays if a part was not present.
        """
        zero_part = np.zeros((len(parts[0])), dtype=np.int32)
        indices = np.argsort(means)[::-1]
        if self.verbose > 0:
            print('Reordering', len(parts), 'parts')
            print('   Means', means)
            print('   Indices', indices)

        if len(parts) == 0:
            result = None
        elif len(parts) == 1:
            result = [parts[0], zero_part, zero_part, zero_part]
        elif len(parts) == 2:
            result = [parts[indices[0]], zero_part, zero_part, parts[indices[1]]]
        elif len(parts) == 3:
            result = [parts[indices[0]], parts[indices[1]], zero_part, parts[indices[2]]]
        elif len(parts) == 4:
            result = [parts[indices[0]], parts[indices[1]], parts[indices[2]], parts[indices[3]]]
        else:
            result = [parts[indices[0]], parts[indices[1]], parts[indices[2]], parts[indices[-1]]]

        new_means = np.zeros(self.required, dtype=np.float32)
        for i, res in enumerate(result):
            new_means[i] = np.mean(result[i])

        if self.verbose > 0:
            print('   New means', new_means)

        return result

    def reduce_time_series(self, parts):
        """ Converts a list of snapshots of notes to changes

        Reduces a matrix with as many rows as time steps of 1/64th into a
        matrix where only changes in time are registered.
        This requires an extra column in which the time at which the
        change took place, will be registered.

        Args:
            parts (list): a list of four array as delivered by reorder_parts with
                 MIDI note values

        Returns:
            a matrix of [number of time changes][5(SATB + time)]
            Each SATB column contains MIDI note values. The last column
            contains the number of time steps to the next change
        """
        parts.append(np.zeros((len(parts[0])), np.int32)) # add extra column for time
        arr = np.transpose(np.array(parts)).astype(dtype=np.int32)
        if self.verbose > 0:
            print('input shape', arr.shape)
        prev = arr[0, :]
        notes = [prev]
        prev_time = 0
        lengte = len(prev) - 1

        for i in range(1, arr.shape[0]):
            if not np.all(arr[i] == prev):
                prev = arr[i, :]
                notes.append(prev)
                duration = i - prev_time
                if duration >= self.n_durs:
                    duration = self.n_durs - 1
                notes[len(notes) - 2][lengte] = duration
                prev_time = i

        return notes

    def ohe_time_series(self, notes, allow_zeros):
        """ Converts MIDI values in to ohe

        converts notes, a matrix created by reduce_time_series, to a one hot
        encoded midi representation.

        Args:
            notes (list): a list of arrays with with each array having
                          5 elements: [SATB + time]

        Returns:
            a list of [SATB + time][n_vocab] matrices, the last
            matrix has shape [SATB + time][n_durs]
        """
        time_series = []
        n_rows = len(notes)
        n_cols = len(notes[0])
        for part in range(n_cols):
            if self.verbose > 1:
                print('Part', part)
            series = []

            # the last part containes time info
            if part < n_cols - 1:
                n_ohe = self.n_vocab
            else:
                n_ohe = self.n_durs

            for t in range(n_rows):
                idx = int(notes[t][part])
                ohe = np.zeros((n_ohe), dtype=np.uint8)

                if idx > 0 or allow_zeros:
                    ohe[idx] = 1

                series.append(ohe)

            time_series.append(np.array(series).astype(dtype=np.uint8))

        return time_series

    def get_notes_as_time_series(self, score, transposing, allow_zeros):
        """ Creates a matrix with SATB+time for each part in score

        Create for each score a list of notes, represented
        as a list of one hot encoded MIDI values. Does the following:
        - get a list of parts, each part is an array of length(n) where
          n is the maximum time steps in 1/64 for the overall score.
          Each part is created by get_time_series_from_part
        - use reorder_parts to create a list of SATB arrays in that order
        - reduce_time_series registers changes of notes, requires an
          extra time array for durations (total is SATB + duration)
        - when convert_to_ohe == True: convertes each array to a OHE matrix
          by ohe_time_series

        Args:
            score (music21 object): a list of 5 matrices
            transposing (bool): True - notes will be transposed to C
            allow_zeros (bool): True - when converting to, allow arrays
                with only zeros to represent zero
            convert_to_ohe (bool):

        Returns:
            list of list of notes
        """

        maxtime = self.compute_maxlen(score)
        if self.verbose > 0:
            print('maxtime =', maxtime)
        if transposing:
            semi_tones = score.mld_semi_tones
        else:
            semi_tones = 0
        means = []
        parts = []
        for part in score.parts:
            n, mean = self.get_time_series_from_part(part, maxtime, semi_tones)
            n_notes = np.count_nonzero(n)
            if self.verbose > 0:
                print('  ', n_notes, 'notes counted')

            # Check whether there are at least minimum_length notes
            if n_notes >= self.minimum_length:
                parts.append(n)
                means.append(mean)

        if self.verbose > 0:
            print(len(parts), 'parts generated')

        # if there is at least one part convert to list with OHE matrices
        if len(parts) > 0:
            parts = self.reorder_parts(parts, means)
            notes = self.reduce_time_series(parts)

            # Function ohe_time_series transposes the notes
            # when not using this function, the notes should be still transposed
            # hence the else part
            notes = list(np.transpose(notes))

            return notes

        #if there are no notes, None is returned
        return None

    def generate_sequences_per_score(self, notes, sequence_length=None):
        """ Creates sequences of notes of a score

        Creates for the notes of a score sequences of sequence_length input (X)
        Yields 5 lists of lists with sequences for X.
        Each sequence is a matrix with sequence_length rows and n_vocab columns.

        Args:
            notes (list of 5 vectors): each vector of shape [n_obs]
            sequence_length (int): the length of each sequence

        Returns:
            a list of 5 lists of n matrices of [sequence_length][n_vocab]
            shape. The number of sequences is depends on the number
            of sequences that can be extratcted from the input matrix
            of a given length
            Returns X as a result when the number of notes is
            greater the the sequence_length, else None, None
        """

        if sequence_length is None:
            sequence_length = self.sequence_length

        # Check whether sequences can be generated
        if len(notes[0]) > sequence_length:
            X = []
            # create input sequences and the corresponding outputs
            # when len(notes.shape) == 1 then each notes vector is one dimensional
            for voice_index, sequence in enumerate(notes):
                x = []
                for i in range(0, len(sequence) - sequence_length, 1):
                    sequence_in = sequence[i:i + sequence_length]
                    x.append(sequence_in)

                X.append(np.array(x))

            return X

        return None # if no sequences can be generated

    def generate_all_sequences(self, scores, transposing, allow_zeros, title, logfile=None):
        """ Generates sequences for all scores

        Generates sequences for all scores and returns these as a list
        of SATB+time matrices (cubes for X, matrices for Y).

        Args:
            scores (list): list of scores to be sequenced

        Returns:
            SATB + time list with cubes of shape
                [n_obs, self.sequence_length, self.n_vocab] as input X
            SATB + time with matrices of shape [n_obs, self.sequence_length]
                 as output Y
        """
        #self.verbose = 1
        prev = 0
        index = {}
        if logfile is not None:
            logfile.p('Sequencing ' + str(len(scores)) + ' scores for ' + title, 'h4')
        X = None

        # TODO: will not work when logfile is None. Should make some provisions for that
        with logfile.Table(['Seq', 'Length', 'From', 'To', 'Sequences', 'Score'],
                            align=['right', 'right', 'right', 'right', 'right', 'left'],
                            hcolors=['gray', 'black']) as tab:
            for score_idx, score in enumerate(scores):
                notes = self.get_notes_as_time_series(score, transposing, allow_zeros)
                if self.verbose > 0:
                    print('  -', score_idx, score.mld_filename)
                if notes is None:
                    tab.row(['', '* No notes in score *', '', '', '', score.mld_filename])
                    continue

                for element in notes:
                    if self.verbose > 0:
                        print(element.shape)

                if X is None:
                    X = self.generate_sequences_per_score(notes, self.sequence_length)
                else:
                    XX = []
                    X_temp = self.generate_sequences_per_score(notes, self.sequence_length)
                    if X_temp is not None:
                        for x, x_temp in zip(X, X_temp):
                            xx = np.concatenate((x, x_temp))
                            XX.append(xx)

                        X = XX

                if X is not None:
                    n = len(X[0])
                    try:
                        if logfile is not None:
                            #print(' -{:5d} -{:6d} [{:6d} -{:5d}]:{:5d} {:s}'.
                            #      format(score_idx, len(notes[0]), prev, n-1, n-prev,
                            #             score.mld_filename), file=logfile)
                            tab.row([str(score_idx), str(len(notes[0])), str(prev),
                                     str(n-1), str(n-prev), score.mld_filename])
                        index[score.mld_filename] = (prev, n-1)
                    except Exception as err:
                        print('exception', err)

                    prev = n
                    if self.verbose > 0:
                        print('Current n:', n)

                    temp = np.array(X[0])
                    if self.verbose > 0:
                        print('Sequence array', temp.shape)
                else:
                    print('   *** X is None - this score not added to sequences')
            # for
        if logfile is not None:
            logfile.flush()

        if X is None:
            return None, None

        XX = []
        for x in X:
            XX.append(np.array(x).astype(dtype=np.uint8))

        return XX, index

    @staticmethod
    def get_one_sequence(X, Y, index):
        x = []
        y = []

        for xi, yi in zip(X, Y):
            xx = xi[index]
            yy = yi[index]

            x.append(xx)
            y.append(yy)

        return x, y

    @staticmethod
    def invert_from_ohe(vector):

        vec = []
        for v in vector:
            mat = np.argmax(v, axis=-1)
            vec.append(mat)

        return vec

    ################### Chord sequence analysis ###########

    def chord_normalize(self, chord):
        a_chord = np.array(chord)
        m = min(a_chord)
        while m < 60:
            a_chord += 12
            m = min(a_chord)

        while m > 71:
            a_chord -= 12
            m = min(a_chord)

        return list(a_chord)

    @staticmethod
    def mp_ccs(notes):
        try:
            if len(notes) > 0:
                c = m21.chord.Chord(notes)
                new_notes = [x + 60 for x in c.normalOrder]

                return (tuple(notes), tuple(new_notes))
        except:
            pass

        return (None, None)

    @staticmethod
    def create_chord_sequence(X, start, end, verbose=0):
        """ Creates a sequence of chords for each score in data X

        Args:
            X (list): List of array containing the sequences
            indices (dict): dictionary containing the start and end value
                of each score in the sequence data

        Returns:
            two lists: the first one contains the original chords,
            the second one the closed chord transposed to octave 4.
        """
        chords_original = []
        chords_closed = []

        source = []

        try:
            x_n = X[0].shape[1]
            for i in range (x_n):
                notes = []
                for j in range(len(X) - 1):
                    note = int(X[j][start][i])
                    if note > 0:
                        notes.append(note)

                source.append(notes)

            for i in range(start + 1, end):
                notes = []
                for j in range(len(X) - 1):
                    note = int(X[j][i][-1])
                    if note > 0:
                        notes.append(note)

                source.append(notes)

            #print('source len', len(source))

            #with mp.Pool() as pool:
            results = list(map(MusicData.mp_ccs, source))

            for (origin, closed) in results:
                if origin is not None:
                    chords_original.append(origin)
                    chords_closed.append(closed)

        except Exception as err:
            print('Error:', str(err))
            chords_original, chords_closed = None, None

        return chords_original, chords_closed

    def create_chord_transitions(self, chords):
        """ Creates chord transitions from data

        Each chord transition is represented as two tuples. The first one contains
        the notes of the first chord, the next tuple of the next chord.
        This transition is stored as the key in a dictionary with value
        being the amopunt of occurrences of this transition

        Args:
            chords (list): sequence of chords created by create_chord_transitions

        Returns:
            a set with chord transitions

        """
        transitions = set(map(lambda i: (chords[i], chords[i+1]), range(len(chords) - 1)))

        return transitions

    def create_transitions(self, X, indices=None, verbose=0, sleutel=''):
        sec = time.time()
        transitions_org = set()
        transitions_cls = set()

        if indices is None:
            indices = {'All': [0, len(X[0])]}

        for key, value in indices.items():
            [start, end] = value

            if verbose > 0:
                if indices is None:
                    print(sleutel, end - start + 1, 'sequences (indices is None)')
                else:
                    print(key, end - start + 1, 'sequences')

            chords_org, chords_cls = self.create_chord_sequence(X, start, end, verbose=0)
            if chords_org is not None and chords_cls is not None:
                trans_org = self.create_chord_transitions(chords_org)
                trans_cls = self.create_chord_transitions(chords_cls)

                for key in trans_org:
                    transitions_org.update(key)
                for key in trans_cls:
                    transitions_cls.update(key)
        # for

        sec = time.time() - sec
        #print(' {:.2f} secs for transitions'.format(sec))

        return transitions_org, transitions_cls

    def extract_one_score(self, X, indices, key):
        sequences = []
        for j in range(len(X)):
            sequences.append([])

        value = indices[key]
        start = value[0]
        end = value[1]

        for i in range(start, end):
            for j in range(len(X)):
                sequences[j].append(X[j][i])

        result = []
        for i in range(len(X)):
            result.append(np.array(sequences[i]))

        return result

    def compute_transition_match(self, transitions, score):
        n_match = len(score & transitions)
        n_count = len(score)

        if n_count < 1:
            return 0
        else:
            return n_match / n_count

    """
    def compute_transition_match(self, transitions, score, key):
        n_match = 0
        n_count = 0

        for key, val in score.items():
            n_count += 1
            if key in transitions:
                n_match += 1
            #else:
            #    if 'dunstable' in key:
            #        print('***', key)

        if n_count == 0:
            return 0, 1

        return n_match / n_count
    """
    ######### Reading and Writing files #############

    def create_sequence_file(self,
                             scores,
                             filename,
                             fractions,
                             transposing,
                             allow_zeros,
                             logfile):
        """ Creates training, validation and test sets based on scores

        Sets are based on scores and not on sequences in order to provide data
        leakage. Converts the scores to sequences and saves these to filename.

        Args:
            scores (list of scores): list of scores that will be partitioned
            indices (list of dict): list dictionaries for each data set where
            filename (str): name of file to save the data sets in
            fractions (list of two floats): [training set fraction, validation
                       set fraction]. if both fractions < 1, the test set fraction
                       will be created as 1 - training fraction - val fraction.
                       When the som > 1 an exception will be raised.
            transposing (bool): when True all notes will be transposed to C major
            allow_zeros (bool): when True zeros are allowed to be sequenced
            convert_to_ohe (bool): when True convert sequences to ohe
            logfile (file handle): handle to logfile to report results

        Raises:
            ValueError: when the sum of fractions > 1

        Returns:
            nothing.
        """
        def save_data_set(scores, idx, start, end, x_name, handle, title, filename):
            lengte = 0
            if start < end:
                score_list = []
                #print(idx)
                for i in idx[start:end]:
                    score_list.append(scores[i])

                print('\n* Writing', title, len(score_list), ' scores as',
                      x_name, 'to', filename)
                logfile.p('Writing ' + title + ' ' + str(len(score_list)) +
                          ' scores as ' + x_name + ' to ' + filename)
                X, index = self.generate_all_sequences(score_list, transposing,
                                            allow_zeros, title, logfile)
                dsetX = handle.create_dataset(x_name, data=X)
                print(dsetX)
                dsetX.attrs['Index'] = json.dumps(index)
                print(index)

                lengte = X[0].shape[0]

                # Create transitions
                transitions_org, transitions_cls = \
                    self.create_transitions(X, indices=index)
                o_trans = yaml.dump(transitions_org)
                c_trans = yaml.dump(transitions_cls)
                #dsetT = handle.create_dataset(x_name + '_transitions', data=s_trans)
                dsetX.attrs['Original Transitions'] = o_trans
                dsetX.attrs['Closed Transitions'] = c_trans
                print('Transitions created')

            return lengte

        train_size = fractions[0]
        if train_size > 1.0 or train_size < 0:
            train_size = 1.0
        val_size = fractions[1]
        test_size = 1.0 - train_size - val_size
        if test_size < 0:
            test_size = 0
            val_size = 1.0 - train_size

        """
        n = len(scores)
        train_size = int(train_size * n)
        val_size = int(train_size + val_size * n)
        if test_size > 0:
            test_size = int(val_size + test_size * n)
        """
        n = len(scores)
        train_size = int(train_size * n)
        if test_size > 0:
            test_size = int(test_size * n)
        val_size = int(n - train_size - test_size)
        val_size += int(train_size)
        test_size += int(val_size)
        print('Final sizes are (train/val/test):', train_size, val_size, test_size)

        idx = np.random.permutation(n).astype(np.int)

        logfile.p('randomizing score over sets', 'h3')
        with h5py.File(filename, mode='w') as hdf5_file:
            # Create train scores
            train_len = save_data_set(scores, idx, 0, train_size,
                                      'X_train', hdf5_file, 'Training set', filename)
            # Validation scores
            val_len = save_data_set(scores, idx, train_size, val_size,
                                    'X_val', hdf5_file, 'Validation set', filename)
            # Test scores
            test_len = save_data_set(scores, idx, val_size, test_size,
                                     'X_test', hdf5_file, 'Test set', filename)
        if test_size > 0: test_size -= val_size
        if val_size > 0: val_size -= train_size

        return [train_size, val_size, test_size], [train_len, val_len, test_len]

    def create_multi_label_output(self, Y):
        """ Stacks all outputs Y as a multi label output set

        Args:
            Y (list): list of input matrices of shape[n_obs, self.sequence_length]

        Returns:
            matrix of shape [n_obs, 4*self.n_vocab + self.n_durs]
        """
        size = 0
        for y in Y:
            size += len(y[0])

        stacked = np.zeros((len(Y[0]), size), dtype=np.uint8)
        offset = 0
        if self.verbose > 0:
            print(stacked.shape)
        for y in Y:
            size = y.shape[1]
            if self.verbose > 0:
                print(offset, size, y.shape)
            #stacked[:, offset:size] = y
            for row in range(y.shape[0]):
                for col in range(offset, offset + size):
                    stacked[row, col] = y[row, col - offset]
            offset += size

        return stacked

    def create_single_input_output(self, X, Y):
        """ Lumps all in put and output together as a single stream.

        Duration data is thrown away, only the sequence of notes counts

        Args:
            X (list): list of cubes of input data
            Y (list): list of matrices of output data

        Returns:
            single cube of input data
            single matrix of output data
        """
        XX = []
        YY = []
        i = 1
        for x, y in zip(X[:len(X)-1], Y[:len(Y)-1]):
            print('-', i)
            i += 1
            for xx, yy in zip(x, y):
                if np.sum(xx) > 0:
                    XX.append(xx)
                    YY.append(yy)

        XX = np.array(XX)
        YY = np.array(YY)

        return XX, YY

    @staticmethod
    def list_sequences(filename):
        with h5py.File(filename, mode='r') as hdf5_file:
            for key in hdf5_file.keys():
                print(key)

        return

    @staticmethod
    def descend_obj(obj, sep='\t'):
        """
        Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
            for key in obj.keys():
                print(sep,'-',key,':',obj[key])
                MusicData.descend_obj(obj[key],sep=sep+'\t')
        elif type(obj)==h5py._hl.dataset.Dataset:
            for key in obj.attrs.keys():
                print(sep+'\t','-',key,':',obj.attrs[key])

    @staticmethod
    def h5dump(path, group='/'):
        """
        print HDF5 file metadata

        group: you can give a specific group, defaults to the root group
        """
        with h5py.File(path,'r') as f:
             MusicData.descend_obj(f[group])

    @staticmethod
    def write_sequences(filename, X, index):
        with h5py.File(filename, mode='w') as hdf5_file:
            dsetX = hdf5_file.create_dataset("X", data=X)

        print(dsetX)

        return

    @staticmethod
    def read_sequences(filename, x_only=False):
        idx_train, idx_val, idx_test = None, None, None
        with h5py.File(filename, mode='r') as hdf5_file:
            X_train = list(hdf5_file["X_train"][:])
            idx_train = hdf5_file['X_train'].attrs['Index']

            try:
                X_val = list(hdf5_file["X_val"][:])
                idx_val = hdf5_file['X_val'].attrs['Index']
            except:
                X_val = None

            try:
                X_test = list(hdf5_file["X_test"][:])
                idx_train = hdf5_file['X_test'].attrs['Index']
            except:
                X_test = None

        data = [X_train, X_val, X_test], [idx_train, idx_val, idx_test]

        return data

    @staticmethod
    def read_transitions(filename, x_only=False):
        with h5py.File(filename, mode='r') as hdf5_file:
            temp_o = hdf5_file['X_train'].attrs['Original Transitions']
            temp_c = hdf5_file['X_train'].attrs['Closed Transitions']
            X_train_transitions = [yaml.load(temp_o), yaml.load(temp_c)]

            try:
                temp_o = hdf5_file["X_val"].attrs['Original Transitions']
                temp_c = hdf5_file["X_val"].attrs['Closed Transitions']
                X_val_transitions = [yaml.load(temp_o), yaml.load(temp_c)]
            except:
                X_val_transitions = None

            try:
                temp_o = hdf5_file["X_test"].attrs['Original Transitions']
                temp_c = hdf5_file["X_test"].attrs['Closed Transitions']
                X_test_transitions = [yaml.load(temp_o), yaml.load(temp_c)]
            except:
                X_test_transitions = None

        data = [X_train_transitions, X_val_transitions, X_test_transitions]

        return data

    @staticmethod
    def ohe(matrix, n):
        cube = np.zeros((matrix.shape[0], matrix.shape[1], n))

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                cube[row, col, matrix[row, col]] = 1

        return cube

    @staticmethod
    def ohe_data(data):
        results =  []
        for matrix in data:
            result = MusicData.ohe(matrix, MusicData.n_vocab)
            results.append(result)

        return results

    @staticmethod
    def un_ohe_voice(voice, normalize):
        if voice is not None:
            if normalize:
                divisor = voice.shape[-1] / 2.0
                result = np.argmax(voice, axis=-1) / divisor - 1.0
            else:
                result = np.argmax(voice, axis=-1).astype('int32')

            return result

        return None

    @staticmethod
    def un_ohe_voices(voices, normalize=False):
        if voices is not None:
            results = []
            for voice in voices:
                result = MusicData.un_ohe_voice(voice, normalize)
                results.append(result)

            return results

        return None

    @staticmethod
    def un_ohe_data(data, normalize):
        """ create floating point data from ohe input

        Ohe data is first converted to integers using numpy.argmax,
        next it is normalized to values between -1 and 1.

        Args:
            data (list of list of arrays):
                - The first list contains:
                    [X_train, Y_train, X_val, Y_val, X_test, Y_test]
                - each element of the list is a list of 5 elements:
                    [array[soprano], alto, tenor, bass, durations]
                - Each X array is of shape (n, sequence length, 128)
                  Each Y array is of shape (n, 128)

                The ?_train elements will always be provided, the other
                may be None.
            normalize(bool): True - data will be normalized between [-1, 1]
                             False - data will be just integers

        Returns:
            The same list/array structure but for each array the last index (-1)
            (of size 128) has been removed and the index before that (-2)
            contains the argmax values normalized between -1 and 1
        """
        results = []
        for voices in data:
            result = MusicData.un_ohe_voices(voices, normalize)
            results.append(np.array(result))

        #results = list(map(MusicData.un_ohe_voices, data))
        return results

    @staticmethod
    def create_y_from_x(X):
        if X is not None:
            if isinstance(X, list):
                x_list = []
                y_list = []
                for x in X:
                    yy = x[:, -1:, :]
                    xx = x[:, :-1, :]
                    x_list.append(xx)
                    y_list.append(yy)

                return x_list, y_list
            else:
                Y = X[:, -1:, :]
                X = X[:, :-1, :]

                return X, Y

        return None, None

    @staticmethod
    def reshape_x(X):
        if X is not None:
            if isinstance(X, list):
                x_list = []
                for x in X:
                    xx = x.reshape((x.shape[0], x.shape[1], 1))
                    x_list.append(xx)

                return x_list
            else:
                X = X.reshape((X.shape[0], X.shape[1], 1))

                return X

        return None

    @staticmethod
    def reshape_y(Y):
        if Y is not None:
            if isinstance(Y, list):
                y_list = []
                for y in Y:
                    yy = y.reshape((y.shape[0], y.shape[2]))
                    y_list.append(yy)

                return y_list
            else:
                Y = Y.reshape((Y.shape[0], Y.shape[2]))

                return Y

        return None

    @staticmethod
    def normalize(arr, range, min, max):
        new_range = max - min
        result = arr / range
        result = result * new_range
        result = result + min

        return result