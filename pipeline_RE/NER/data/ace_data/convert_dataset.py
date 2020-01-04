"""Script to convert data set format"""

__author__ = "Xinle Wu"

from pathlib import Path

if __name__ == '__main__':

    def filename(name):
        return '{}.txt'.format(name)

    words = []
    words_list = []
    tags = []
    tags_list = []
    for n in ['train', 'dev', 'test']:
        with Path(filename(n)).open(encoding='utf-8') as fr:
            for line in fr:
                if line != '=============\t=============\t=============\n':
                    fileds = line.strip().split('\t')
                    if len(fileds) != 3:
                        continue
                    words.append(fileds[0])
                    tags.append(fileds[1])
                else:
                    if len(words) > 3:  # 长度不足4的句子全部丢弃
                        words_list.append(' '.join(words))
                        tags_list.append(' '.join(tags))
                    words = []
                    tags = []
        with Path(filename(n + '.words')).open('w', encoding='utf-8') as fw:
            for w in words_list:
                fw.write('{}\n'.format(w))
        with Path(filename(n + '.tags')).open('w', encoding='utf-8') as fw:
            for t in tags_list:
                fw.write('{}\n'.format(t))
        words_list = []
        tags_list = []
