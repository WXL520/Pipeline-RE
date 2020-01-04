"""bi_LSTM for Relation Classification"""

__author__ = "Xinle Wu"

import logging
import functools
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup, rnn
from tf_metrics import precision, recall, f1

# tf.enable_eager_execution()  # 具体用法还不熟悉

DATADIR = '../../data/pubmed_data'

# logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)  # 记录INFO级别以上的消息
handlers = [
    logging.FileHandler('results/main.log'),  # 打印日志信息到磁盘文件的句柄
    logging.StreamHandler(sys.stdout)  # 打印日志信息到标准输出的句柄
]
logging.getLogger('Tensorflow').handlers = handlers  # 初始化一个名为TensorFlow的记录器


def parse_fn(line_words, line_tag):
    # When feeding string objects to your graph, you need to encode your string to bytes.
    words = [w.encode() for w in line_words.strip().split()]
    tag = [t.encode() for t in line_tag.strip().split()]
    assert len(tag) == 1, "The length of tag must be 1"
    return (words, len(words)), tag[0]


def generator_fn(words, tags):
    with Path(words).open(encoding='utf-8') as f_words, Path(tags).open(encoding='utf-8') as f_tags:
        for line_words, line_tag in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tag)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), ())  # 对应generator返回的每一个元素
    types = ((tf.string, tf.int32), tf.string)  # 这里类型和形状的定义不太确定
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(  # 推荐的数据导入方式
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types
    )

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epoch'])

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
    words, nwords = features  # features对应input_fn返回的第一项
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as fr:
        indices = [idx for idx, tag in enumerate(fr) if tag.strip() != 'Other']
        num_tags = len(indices) + 1

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    var = np.vstack([glove, [[0.] * params['dim']]])  # for unknown words
    var = tf.Variable(initial_value=var, dtype=tf.float32, trainable=False)  # 不允许对glove词向量进行微调
    embeddings = tf.nn.embedding_lookup(var, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)  # 只在训练时dropout

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)  # 调用类方法
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    # output = tf.layers.dropout(output, rate=dropout, training=training)

    # Attention
    if params['attention']:
        a = 0
    else:
        output = tf.reduce_max(output, axis=1)  # 最大池化
    output = tf.layers.dropout(output, rate=dropout, training=training)
    logits = tf.layers.dense(output, num_tags)
    pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = lookup.index_to_string_table_from_file(params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        one_hot_tags = tf.one_hot(tags, depth=num_tags)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(None, one_hot_tags, logits)
        loss = tf.reduce_mean(loss)

        # Metrics
        weights = tf.sequence_mask(nwords)  # weights参数起什么作用啊？
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids),
            # 这里应该将'Other'视为负类？
            'precision': precision(tags, pred_ids, num_classes=num_tags, pos_indices=indices),
            'recall': recall(tags, pred_ids, num_classes=num_tags, pos_indices=indices),
            'f1': f1(tags, pred_ids, num_classes=num_tags, pos_indices=indices)
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
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,  # 指定oov单词的id，1表示设为词汇表大小
        'epoch': 40,
        'batch_size': 64,
        'buffer': 500000,  # buffer_size需要大于Dataset中样本的数量？确保数据完全被随机化处理
        'lstm_size': 300,
        'attention': False,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w') as fw:
        json.dump(params, fw, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('test'), ftags('test'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)  # 每隔两分钟保存一次检查点
    estimator = tf.estimator.Estimator(model_fn, model_dir='results/model', config=cfg, params=params)  # config可以设置cpu核数使用情况
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)  # hook假定eval_dir一定存在，所以需要先创建
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)  # early-stopping,500个step内f1值没有提升，就停止训练
    train_spec = tf.estimator.TrainSpec(train_inpf, hooks=[hook])
    # train_spec = tf.estimator.TrainSpec(train_inpf)
    eval_spec = tf.estimator.EvalSpec(eval_inpf, throttle_secs=120)  # 不需要迭代一轮就能评估？train和eval并行？
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb', encoding='utf-8') as fw:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            golds_gen = generator_fn(fwords(name), ftags(name))
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                # fw.write(b' '.join([' '.join(words), tags, preds['tags']]) + b'\n')
                fw.write(b' '.join(words) + b'\t' + tags + b'\t' + preds['tags'] + b'\n')
                fw.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)

# dataset = input_fn(DATADIR + '/test.words.txt', DATADIR + '/test.tags.txt')
# iterator = dataset.make_one_shot_iterator()
# node = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run(node))
#     print(sess.run(node[0][0]))
#     print(type(node[0][0]))