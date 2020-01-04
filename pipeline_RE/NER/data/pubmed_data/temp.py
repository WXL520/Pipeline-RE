from pathlib import Path

# new_lines = []
# with Path('origin_train.txt').open(encoding='utf-8') as fr:
#     for line in fr:
#         fields = line.strip().split('\t')
#         word = fields[0].split()
#         if len(fields) != 3 or len(word) != 1:
#             continue
#         new_line = '\t'.join(fields)
#         new_lines.append(new_line)
# with Path('train.txt').open('w', encoding='utf-8') as fw:
#     for line in new_lines:
#         fw.write(line + '\n')

num = 0
with Path('dev.words.txt').open(encoding='utf-8') as f1, Path('dev.tags.txt').open(encoding='utf-8') as f2:
    for line1, line2 in zip(f1, f2):
        words = [l for l in line1.strip().split()]
        tags = [l for l in line2.strip().split()]
        if len(words) != len(tags):
            num += 1
            print(line1, line2)
print(num)