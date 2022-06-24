# coding=utf-8
#__author__ = 'yjxiong'

import os
import glob
import sys
from pipes import quote
from multiprocessing import Pool, current_process   # 多进程管理包
import argparse
import datetime
out_path = ''       # 定义的是一个全局变量

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# 测试实时更新


def dump_frames(vid_path):      # 这个函数在这里确实没有调用
    import cv2
    video = cv2.VideoCapture(vid_path)      # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
    vid_name = vid_path.split('/')[-1].split('.')[0]        # 获取视频文件的额名字
    out_full_path = os.path.join(out_path, vid_name)        # 视频输出的位置以及输出的名字，生成保存每个视频图片的子文件夹

    fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))     # 获取视频总的帧数
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(fcount):
        ret, frame = video.read()       # 按帧读取视频，ret是bool值，读取帧正确返回true，文件读到结尾返回False，frame就是每一帧的图像，是个三维矩阵。
        assert ret
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()      # 用于多线程刷新
    return file_list

# LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
def run_optical_flow(vid_item, dev_id=0, current_video_order=0):
    current_video_order = current_video_order + 1
    vid_path = vid_item[0]      # vid_item是一个二维的元组，0是视频的绝对路径，1是对所有视频调用线程时的一个顺序标记
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]        # -1应该是表示取分割后的最后一个元素
    action_name = vid_path.split('/')[-2]
    #out_full_path = os.path.join(out_path, vid_name)        # out_path是一个全局变量，所以此处可以使用，os.path.join()连接两个或更多的路径名组件
    out_full_path = os.path.join(out_path + action_name + '/', vid_name)
    try:
        os.makedirs(out_full_path)
        #os.mkdir(out_full_path)     # 生成文件夹，os.mkdir() 方法用于以数字权限模式创建目录。默认的模式为 0777，如果已经存在则不会被覆盖
    except OSError:
        pass

    current = current_process()     # current_process()获取代表当前进程的全局变量
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU      # 当前使用的总的进程数，-1表示目前已经使用了一个进程，剩下的进程除以GPU的数量取余数，设置为GPU的号数，一个进程可以调用一块GPU
    image_path = '{}/img'.format(out_full_path)     # 就算有文件夹生成的文件也没有放在里面,{}/img是同级目录
    #os.makedirs(image_path)                        # {}/flow_x/  就会把文件存放于flow_x文件夹下
    flow_x_path = '{}/flow_x'.format(out_full_path)
    #os.makedirs(flow_x_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)
    #os.makedirs(flow_y_path)
    #print(os.path.join(df_path + 'build/extract_gpu'))
    # 前向光流计算
    # cmd = os.path.join(df_path + 'build/extract_gpu')+' -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o={} -w={} -h={}'.format( # df_path表示使用的提取光流的可执行文件所在的位置
    #     quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

    # 后向光流计算
    cmd = os.path.join(df_path + 'build_backward/extract_gpu')+' -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o={} -w={} -h={}'.format( # df_path表示使用的提取光流的可执行文件所在的位置
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])

    os.system(cmd)
    print('{} {} done {}/{}'.format(vid_id, vid_name, current_video_order, video_total_frame ))
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
    with open('/home/ange/project-2022/tsn-20200722-output/bulf_0f-recorder.txt', 'a') as f:
        f.write('{} {} done {}/{}'.format(vid_id, vid_name, current_video_order, video_total_frame ) + '    ' + time_str + '\n')
    sys.stdout.flush()        # sys.stdout.flush()，刷新stdout ，可以用在网络程序中多线程程序，多个线程后台运行，同时要能在屏幕上实时看到输出信息。
    return True


def run_warp_optical_flow(vid_item, dev_id=0):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    cmd = os.path.join(df_path + 'build/extract_warp_gpu')+' -f {} -x {} -y {} -b 20 -t 1 -d {} -s 1 -o {}'.format(
        vid_path, flow_x_path, flow_y_path, dev_id, out_format)

    os.system(cmd)
    print('warp on {} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True

def nonintersection(lst1, lst2):
    lst3 = [value for value in lst1 if ((value.split("/")[-1]).split(".")[0]) not in lst2]
    return lst3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    # parser.add_argument("src_dir",type=str, default='/home/ange/projects/temporal-segment-networks/tools/UCF101/')
    # parser.add_argument("out_dir", type=str, default='/home/ange/projects/temporal-segment-networks/tools/images/')
    parser.add_argument("--src_dir",type=str, default='/home/ange/projects/temporal-segment-networks/tools/images/new_videos/')
    parser.add_argument("--out_dir", type=str, default='/home/ange/projects/temporal-segment-networks/tools/images/new_images/')
    parser.add_argument("--num_worker", type=int, default=8)        # 默认值是8
    parser.add_argument("--flow_type", type=str, default='tvl1', choices=['tvl1', 'warp_tvl1'])
    #parser.add_argument("--df_path", type=str, default='./lib/dense_flow/', help='path to the dense_flow toolbox')      # ./表示当前文件夹，即build_of.py文件所在的文件夹
    # parser.add_argument("--df_path", type=str, default='../dense_flow/', help='path to the dense_flow toolbox')  # 计算光流的相对路径
    parser.add_argument("--df_path", type=str, default='/home/ange/projects/temporal-segment-networks/dense_flow/', help='path to the dense_flow toolbox')  # 计算光流工具的绝对路径

    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi','mp4'], help='video file extensions')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_gpu", type=int, default=4, help='number of GPU')     # 默认值是8
    parser.add_argument("--resume", type=str, default='yes', choices=['yes','no'], help='resume optical flow extraction instead of overwriting')# 恢复光流提取而不是覆盖,默认为no

    args = parser.parse_args()
    print(os.getcwd())

    complete_act = ['FieldHockeyPenalty', 'JugglingBalls', 'TaiChi']
    print(complete_act)
    out_path = args.out_dir     # out_path表示提取的图片的输出位置
    src_path = args.src_dir     # src_path表示视频文件存储的位置
    num_worker = args.num_worker        # num_worker表示调用的线程数
    flow_type = args.flow_type          # 表示提取光流的形式是tvl1还是warp_tvl1
    df_path = args.df_path              # 表示denseflow工具箱的位置，即提取光流的可执行文件的位置
    out_format = args.out_format        # 输出文件格式，是dir的文件夹还是zip类型的压缩文件
    ext = args.ext                      # 视频文件的扩展名，是.avi文件，还是MP4文件
    new_size = (args.new_width, args.new_height)    # 新文件的大小
    NUM_GPU = args.num_gpu              # 调用GPU的数量
    resume = args.resume                # 恢复光流的提取而不是覆盖
    if not os.path.isdir(out_path):     # 如果输出的位置不是一个文件夹执行
        print("creating folder: "+out_path)     # 输出在out_path处生成一个文件夹
        os.makedirs(out_path)       # 该方法用于递归创建目录，路径上的所有文件夹都会被创建
    print("reading videos from folder: ", src_path)
    print("selected extension of videos:", ext)     # ext表示的是文件的后缀

    com_vid_list = os.listdir(out_path)
    total_video = os.listdir(src_path)
    # remain_list = nonintersection(total_video, com_vid_list)
    # for unextract_root in remain_list:
    #     for root, dirs, files in os.walk(os.path.join(src_path, unextract_root)):
    #         current_file_name = root.split('/')[-1]
    #         vid_list = glob.glob(root+'/*.'+ext)
    #         global video_total_frame
    #         video_total_frame = len(vid_list)
    #         pool = Pool(num_worker)  # num_worker
    #         #current_video_order = 0
    #         if flow_type == 'tvl1':
    #             print('extract flow')
    #             pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
    #         elif flow_type == 'warp_tvl1':
    #             pool.map(run_warp_optical_flow, zip(vid_list, range(len(vid_list))))

    for i in complete_act:  # 将已经提取的动作保存在complete_act，剔除已经提取的动作
        if i in total_video:
            total_video.remove(i)

    for action_name in total_video:  # 这是基于当前文件目录写的提取光流的程序
        vid_list = glob.glob(src_path + action_name + '/*.' + ext)  # 他这里不会按照字母的顺序来进行搜索
        global video_total_frame
        video_total_frame = len(vid_list)
        pool = Pool(num_worker)  # num_worker
        current_video_order = 0
        if flow_type == 'tvl1':
            print('extract flow')
            pool.map(run_optical_flow, zip(vid_list, range(len(vid_list))))
        elif flow_type == 'warp_tvl1':
            pool.map(run_warp_optical_flow, zip(vid_list, range(len(vid_list))))

