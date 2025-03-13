#!/bin/bash

# 检查是否传入路径
if [ -z "$1" ]; then
    echo "Error: No path provided. Please provide a path."
    exit 1
fi

# 接收传入的路径
path=$1
echo "Path: $path"
# 检查是否传入 dynamic 参数，默认值为 false
dynamic="False"
if [ ! -z "$2" ]; then
    dynamic="$2"
fi
echo "Dynamic: $dynamic"
# 检查路径是否存在
if [ ! -d "$path" ]; then
    echo "Error: Provided path does not exist. Please provide a valid path."
    exit 1
fi

# 清空路径下的所有文件夹
echo "Clearing all folders under $path..."
find "$path" -mindepth 1 -type d -exec rm -rf {} +

# 文件夹列表
folders=("elementwise_add" "elementwise_mul" "elementwise_broadcast0_add" "elementwise_broadcast1_add" "elementwise_broadcast1_mul" 
            "silu" "rmsnorm" "layernorm" "transpose01" "transpose23" "split" "vectorAdd" "softmax" "gelu" "linearw8" 
            "linearw8_bias" "convert" "div" "rope" "matmul_int" "misc_transpose12" "elementwise_broadcast2_add" "elementwise_broadcast3_add" "split1")

# 显示文件夹序号和名称的对应关系（横排显示）
echo "Available cases:"
columns=4  # 每行显示的列数

for i in "${!folders[@]}"; do
    # 输出序号和文件夹名称
    printf "%2d) %-30s" $((i + 1)) "${folders[$i]}"
    # 每列数达到指定数量换行
    if (( (i + 1) % columns == 0 )); then
        echo
    fi
done

# 如果最后一行未满，手动换行
if (( ${#folders[@]} % columns != 0 )); then
    echo
fi

# 提示用户是否选择全部文件夹，默认为全部
echo "Do you want to create all cases (default) or select specific cases?"
echo "1) Create all cases"
echo "2) Select specific cases"

read -p "Enter your choice (1 for all, 2 for specific): " choice

# 根据用户输入的选择决定操作
if [ -z "$choice" ]; then
    choice=1  # 默认选择1，即创建所有文件夹
fi

# 如果选择是2，询问用户选择哪些文件夹
selected_folders=()
selected_indices=()
if [ "$choice" -eq 2 ]; then
    echo "Please enter the numbers of the cases to create (e.g., 1 3 5):"
    read -p "Enter cases numbers: " input
    for num in $input; do
        # 检查输入是否合法
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#folders[@]}" ]; then
            selected_folders+=("${folders[$num-1]}")
            selected_indices+=("$num")
        else
            echo "Error: Invalid selection: $num. Exiting..."
            exit 1
        fi
    done
fi

# 如果没有选择任何文件夹，默认选择全部
if [ ${#selected_folders[@]} -eq 0 ]; then
    selected_folders=("${folders[@]}")
    for i in "${!folders[@]}"; do
        selected_indices+=("$((i + 1))")
    done
fi

# # 遍历用户选择的文件夹
# for folder in "${selected_folders[@]}"; do
#     folder_path="$path/$folder"
    
#     if [ -d "$folder_path" ]; then
#         echo "Folder $folder_path exists. Deleting and recreating..."
#         rm -rf "$folder_path"
#     else
#         echo "Folder $folder_path does not exist. Creating..."
#     fi

#     # 创建文件夹
#     mkdir -p "$folder_path"
# done

# echo "All specified folders have been checked and created."

# 运行 Python 脚本，传入选定的序号列表
echo "Running _misc_lower.py with path: $path and selected indices: ${selected_indices[*]}"
# python generate_pvpu_case_real.py "$path" "$dynamic" "${selected_indices[@]}" 
python generate_case.py "$path" "$dynamic" "${selected_indices[@]}" 
archive_path="$path/case.tar.gz"

# 获取传入路径下的所有最外层目录的名称
directories=$(find "$path" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;)

# 如果没有找到任何目录，提示错误
if [ -z "$directories" ]; then
    echo "Error: No directories found in the provided path."
    exit 1
fi

# 使用 tar 命令将这些目录打包
echo "Creating a tar.gz package named 'case.tar.gz' and saving it to $path"
tar -czf "$archive_path" -C "$path" $directories

if [ $? -eq 0 ]; then
    echo "Compression completed. The package is saved as '$archive_path'."
else
    echo "Error: Failed to create the tar.gz package."
    exit 1
fi
