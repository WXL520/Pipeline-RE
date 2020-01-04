from collections import defaultdict
import argparse


def f1(y_true, y_pred):
    d = defaultdict(int)
    for i, j in zip(y_true, y_pred):
        if i == j:
            d[i + '_TP'] += 1
        else:
            d[j + '_FP'] += 1
            d[i + '_FN'] += 1
    TP = 0
    FP = 0
    FN = 0
    for i, j in d.items():
        if i.endswith('_TP') and i != args.OTHER_RELATION + '_TP':
            TP += j
        if i.endswith('_FP') and i != args.OTHER_RELATION + '_FP':
            FP += j
        if i.endswith('_FN') and i != args.OTHER_RELATION + '_FN':
            FN += j
    Pr = TP / (TP + FP + 0.001)
    Rc = TP / (TP + FN + 0.001)
    F1 = (2 * Pr * Rc) / (Pr + Rc + 0.001)
    return '{0:.2f}'.format(Pr * 100), '{0:.2f}'.format(Rc * 100), '{0:.2f}'.format(F1 * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--OTHER_RELATION', default='Other', type=str)
    parser.add_argument('--predict_file', default='./prediction_result.txt', type=str)
    parser.add_argument('--label_file', default='./real_result.txt', type=str)
    args = parser.parse_args()

    s_1, s_2 = [], []
    with open(args.predict_file) as infile:
        for line in infile:
            line = line.strip().split('\t')
            s_2.append(line[-1])
    with open(args.label_file) as infile:
        for line in infile:
            line = line.strip().split('\t')
            s_1.append(line[-1])
    score = f1(s_1, s_2)
    print(float(score[0]), float(score[1]), float(score[2]))

