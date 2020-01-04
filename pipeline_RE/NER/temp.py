from pathlib import Path

correct_num = 0
with Path('test.tags.txt').open() as f_tag, Path('result.txt').open() as f_pred:
    lines_tag = f_tag.readlines()
    lines_pred = f_pred.readlines()
    for i in range(len(lines_tag)):
        tag_list = lines_tag[i].strip().split()
        pred_list = lines_pred[i].strip().split()
        tag_length = len(tag_list)
        # print(tag_list)
        # print(pred_list[:tag_length ])
        # break
        if tag_list == pred_list[:tag_length]:
            correct_num += 1
print(correct_num/len(lines_tag))
