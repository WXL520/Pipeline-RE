__author__ = "Xinle Wu"

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.contrib import predictor

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'The folder dir where the test data is located')
flags.DEFINE_string('output', None, 'The path of output file')

MODEL_DIR = 'saved_model'
# DATADIR = '../../data/pubmed_data/'
batch_size = 1024


def parse_fn(line_words):
    words = [w for w in line_words.strip().split()]
    return words, len(words)
    # chars = [[c for c in w] for w in line_words.strip().split()]
    # nchars = [len(c) for c in chars]
    # max_len = max(nchars)
    # chars = [c + ['<pad>'] * (max_len - l) for c, l in zip(chars, nchars)]

    # return {'words': words, 'nwords': len(words),
    #         'chars': chars, 'nchars': nchars}


def generator_fn(words_path):
    with Path(words_path).open(encoding='utf-8') as fr:
        lines = fr.readlines()

        max_sent_len = 0
        max_char_len = 0
        words_list = []
        nwords_list = []
        chars_list = []
        nchars_list = []
        for i, line_words in enumerate(lines, start=1):
            words, nwords = parse_fn(line_words)
            chars = [[c for c in w] for w in words]
            nchars = [len(c) for c in chars]
            max_len = max(nchars)
            max_char_len = max_len if max_len > max_char_len else max_char_len
            max_sent_len = nwords if nwords > max_sent_len else max_sent_len
            words_list.append(words)
            nwords_list.append(nwords)
            chars_list.append(chars)
            nchars_list.append(nchars)
            if i % batch_size == 0 or i == len(lines):
                batch_words = np.array(
                    [w + ['<pad>'] * (max_sent_len - l) for w, l in zip(words_list, nwords_list)])  # pad sentence
                batch_nwords = np.array([max_sent_len] * len(nwords_list))
                padded_chars = [chars + [['<pad>']] * (max_sent_len - len(chars)) for chars in chars_list]
                padded_nchars = [nchars + [1] * (max_sent_len - len(nchars)) for nchars in nchars_list]
                batch_chars = np.array(
                    [[c + ['<pad>'] * (max_char_len - n) for c, n in zip(chars, nchars)]  # pad words
                     for chars, nchars in zip(padded_chars, padded_nchars)])
                batch_nchars = np.array(padded_nchars)

                words_list = []
                nwords_list = []
                chars_list = []
                nchars_list = []
                max_char_len = 0
                max_sent_len = 0

                yield {'words': batch_words, 'nwords': batch_nwords,
                       'chars': batch_chars, 'nchars': batch_nchars}


def main(_):
    subdirs = [s for s in Path(MODEL_DIR).iterdir()
               if s.is_dir() and 'temp' not in str(s)]
    latest_model = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest_model)

    def fword(name):
        return Path(FLAGS.data_dir, '{}.words.txt'.format(name))

    tic = time.time()
    predictions = []
    for line in generator_fn(fword('test')):
        predictions.extend(predict_fn(line)['tags'])
    toc = time.time()

    with Path(FLAGS.output).open('w') as fw:
        for sent_tag in predictions:
            tags = ' '.join([s.decode() for s in sent_tag])
            fw.write(tags + '\n')
    print("The total time is: {}s".format(toc - tic))


if __name__ == '__main__':
    tf.flags.mark_flag_as_required('data_dir')
    tf.flags.mark_flag_as_required('output')
    tf.app.run()
