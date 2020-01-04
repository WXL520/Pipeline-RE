__author__ = "Xinle Wu"

import time
from pathlib import Path
from tensorflow.contrib import predictor
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'The folder dir where the test data is located')
flags.DEFINE_string('output', None, 'The path of output file')

MODEL_DIR = 'saved_model'
# DATADIR = '../../data/pubmed_data'
batch_size = 2048


def parse_fn(line_words):
    words = [w.encode() for w in line_words.strip().split()]
    return words, len(words)


def generator_fn(words_path):
    with Path(words_path).open(encoding='utf-8') as fr:
        lines = fr.readlines()

        max_len = 0
        words_list = []
        nwords_list = []
        for i, line_words in enumerate(lines, start=1):
            words, nwords = parse_fn(line_words)
            max_len = nwords if nwords > max_len else max_len
            words_list.append(words)
            nwords_list.append(nwords)
            if i % batch_size == 0 or i == len(lines):  # 按batch传入数据
                batch_words = np.array(
                    [w + [b'<pad>'] * (max_len - l) for w, l in zip(words_list, nwords_list)])  # pad sentence
                batch_nwords = np.array(nwords_list)

                words_list = []
                nwords_list = []
                max_len = 0
                yield batch_words, batch_nwords


def main(_):
    subdirs = [s for s in Path(MODEL_DIR).iterdir()
               if s.is_dir() and "temp" not in str(s)]
    latest_model = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest_model)

    def fwords(name):
        return Path(FLAGS.data_dir, '{}.words.txt'.format(name))

    # tic = time.time()
    # predictions = []
    # for batch_words, batch_nwords in generator_fn(fwords('test')):
    #     predictions.extend(
    #         predict_fn({'words': batch_words, 'nwords': batch_nwords})['tags'])
    # toc = time.time()

    tic = time.time()
    predictions = []
    for batch_words, batch_nwords in generator_fn(fwords('test')):
        preds = predict_fn({'words': batch_words, 'nwords': batch_nwords})['tags']
        for pred, n in zip(preds, batch_nwords):
            predictions.append(pred[:n])
    toc = time.time()

    with Path(FLAGS.output).open('w', encoding='utf-8') as fw:
        for line in predictions:
            tags = ' '.join([t.decode() for t in line])
            fw.write(tags + '\n')
    print("The total time is: {}s".format(toc - tic))


if __name__ == '__main__':
    tf.flags.mark_flag_as_required('data_dir')
    tf.flags.mark_flag_as_required('output')
    tf.app.run()