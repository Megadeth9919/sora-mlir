#!/bin/bash

# profiling_inst.sh
# This script is used to profile the instruction.

# Usage: ./profiling_inst.sh inst_dir fig_flag txt_flag
# inst_dir: the directory where the instruction files are stored (0.json, 1.json, 2.json)
# fig_flag: 1 to generate figures, 0 to skip figures
# txt_flag: 1 to generate txt files, 0 to skip txt files

inst_dir=$1
fig_flag=$2
txt_flag=$3

python profiling_inst.py -p $inst_dir -f $fig_flag -t $txt_flag
