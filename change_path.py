# coding=utf-8

# 这个文件的作用是将反向图像的路径进行替换  替换成正确的txt文件的路径
import argparse

# true_path /data/ange/UCF101/images_backforwad/
# --source_file /home/ange/projects/temporal-segment-networks/data/back_ucf101_flow_train_split_1.txt
# --write_file /home/ange/projects/temporal-segment-networks/data/back_new_ucf101_flow_train_split_1.txt
parser = argparse.ArgumentParser(description='生成数据集中视频路径的txt文件')
parser.add_argument('true_path', type=str)

parser.add_argument('--source_file', type=str, default='/home/ange/projects/temporal-segment-networks/data/back_ucf101_'
                                                       'flow_train_split_1.txt')
parser.add_argument('--write_file', type=str, default='/home/ange/projects/temporal-segment-networks/data/back_new_'
                                                      'ucf101_flow_train_split_1.txt')


def overwrite_vid_path(true_path, source_path, write_path):
    # true_path = '/data/ange/UCF101/images_backforwad/'
    with open(source_path) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            action_name = line.split('_')[1]

            new_line = true_path + action_name + '/' + line[25:]
            print(line)
            with open(write_path, 'a') as fw:
                fw.writelines(new_line)
        print('overwrite done!')


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    overwrite_vid_path(args.true_path, args.source_file, args.write_file)



# 修改路径
# /data/ange/UCF101/images_backforwad/
# --source_file
# /home/ange/projects/temporal-segment-networks/data/back_ucf101_flow_train_split_1.txt
# --write_file
# /home/ange/projects/temporal-segment-networks/data/back_new_ucf101_flow_train_split_1.txt

# /data/ange/UCF101/images_backforwad/
# --source_file
# /home/ange/projects/temporal-segment-networks/data/back_ucf101_flow_val_split_1.txt
# --write_file
# /home/ange/projects/temporal-segment-networks/data/back_new_ucf101_flow_val_split_1.txt