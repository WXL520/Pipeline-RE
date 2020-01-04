"""Script to convert data set format"""

__author__ = "Xinle Wu"

from pathlib import Path
from collections import defaultdict

if __name__ == '__main__':

    def filename(name):
        return '{}.txt'.format(name)

    id2tag = defaultdict()
    with Path('relation_type.txt').open() as fr:
        for idx, line in enumerate(fr):
            id2tag[idx] = line.strip()

    words = []
    words_list = []
    tag = []
    tag_list = []
    for n in ['train', 'dev', 'test']:
        with Path(filename(n)).open(encoding='utf-8') as fr:
            for line in fr:
                fileds = line.strip().split()
                words = ' '.join(fileds[1:])
                tag = id2tag[int(fileds[0])]
                words_list.append(words)
                tag_list.append(tag)
        with Path(filename(n + '.words')).open('w', encoding='utf-8') as fw:
            for w in words_list:
                fw.write('{}\n'.format(w))
        with Path(filename(n + '.tags')).open('w', encoding='utf-8') as fw:
            for t in tag_list:
                fw.write('{}\n'.format(t))
        words_list = []
        tag_list = []
