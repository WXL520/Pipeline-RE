from pathlib import Path


pubmed_1 = []
pubmed_2 = []
pubmed_3 = []
pubmed_4 = []
with Path('pubmed_data.txt').open(encoding='utf-8') as fr:
    for i, line in enumerate(fr):
        if i < 9000000:
            pubmed_1.append(line)
        elif 9000000 <= i < 15000000:
            pubmed_2.append(line)
        elif 15000000 <= i < 21000000:
            pubmed_3.append(line)
        else:
            pubmed_4.append(line)

with Path('pubmed_data1.txt').open('w', encoding='utf-8') as fw:
    for line in pubmed_1:
        fw.write(line)
with Path('pubmed_data2.txt').open('w', encoding='utf-8') as fw:
    for line in pubmed_2:
        fw.write(line)
with Path('pubmed_data3.txt').open('w', encoding='utf-8') as fw:
    for line in pubmed_3:
        fw.write(line)
with Path('pubmed_data4.txt').open('w', encoding='utf-8') as fw:
    for line in pubmed_4:
        fw.write(line)
