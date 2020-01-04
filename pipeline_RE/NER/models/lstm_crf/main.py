"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

DATADIR = '../../data/pubmed_data'

# Logging
Path('results').mkdir(exist_ok=True)  # 创建名为results的文件夹
tf.logging.set_verbosity(logging.INFO)  # 记录INFO以上级别的消息
handlers = [
    logging.FileHandler('results/main.log'),  # 打印日志信息到磁盘文件的句柄
    logging.StreamHandler(sys.stdout)  # 打印日志信息到标准输出的句柄
]
logging.getLogger('tensorflow').handlers = handlers  # 初始化一个名为TensorFlow的记录器？


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    # When feeding string objects to your graph, you need to encode your string to bytes.
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r', encoding='utf-8') as f_words, Path(tags).open('r', encoding='utf-8') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(  # 通过tf.data API输入数据
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):  # 如果输入是字典，转成元组格式
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features  # features对应input_fn返回的第一项，words是padded_words，nwords是真实长度
    training = (mode == tf.estimator.ModeKeys.TRAIN)  # 666
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])  # 返回lookup table(word2id)
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    word_ids = vocab_words.lookup(words)  # 将OOV单词和'<pad>'标记都映射为全0向量，这可能在一定程度上影响了模型效果。。
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.]*params['dim']]])  # for unknown words
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)  # 不允许对glove词向量做微调
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # 转置是为了符合对入口参数的要求
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)  # 调用类方法？
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)  # 转移矩阵
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)  # TensorFlow内部解码
    # pred_ids = []
    # for logit, nword in zip(logits, nwords):
    #     pred_ids.append(
    #         tf.contrib.crf.viterbi_decode(logit[:nword], crf_params))  # 外部解码

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:  # 验证集上不优化
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:  # 训练集上优化
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,  # 设为1导致所有oov单词都对应同一个ID，即词汇表大小
        'epochs': 25,
        'batch_size': 64,
        'buffer': 15000,  # buffer_size需要大于Dataset中样本的数量，确保数据完全被随机化处理
        'lstm_size': 300,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),  # 标签种类
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('dev'), ftags('dev'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)  # 检查点保存得太频繁了叭？
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)  # model_fn的参数是怎么传入的？
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)  # early-stopping? 500个step内f1值没有提升，就停止训练
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)  # 不需要迭代一轮就能评估？train和eval并行？
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)
