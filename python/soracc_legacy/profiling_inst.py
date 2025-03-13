# -*-coding:utf-8-*-
import os
import argparse
        
from utils.tools import parse_json_to_inst, load_json
from profiler import InstProfiler


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Convert quantized model to float model')
    parser.add_argument('--inst_path', '-p', type=str, default=None, help='path to quantized weight file')
    parser.add_argument('--dump_fig_flag', '-f', type=str2bool, default=False, help='whether to dump the figure of the profiling result')
    parser.add_argument('--dump_txt_flag', '-t', type=str2bool, default=False, help='whether to dump the txt file of the profiling result')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.inst_path is None:
        print('Please specify the path to the quantized weight file')
        return
    if not os.path.exists(args.inst_path):
        print('The specified path does not exist')
        return
    if not os.path.isdir(args.inst_path):
        print('The specified path is not a directory')
        return
    
    inst_path = os.path.abspath(args.inst_path)

    file_list = os.listdir(inst_path)    
    
    # 把所有json文件提取出来
    json_file_list = [os.path.join(inst_path, file_name) for file_name in file_list if file_name.endswith('.json')]
    json_file_list = sorted(json_file_list, key=lambda x: int(x.split("/")[-1].split(".")[0][-1]))

    inst_dict_list = list()
    for file_name in json_file_list:
        json_data = load_json(file_name)
        inst_dict_list.append(json_data)
    
    inst_list = parse_json_to_inst(inst_dict_list)
    InstProfiler(inst_list).run(dump_breakdown=args.dump_fig_flag, dump_txt=args.dump_txt_flag)

if __name__ == '__main__':
    main()
    