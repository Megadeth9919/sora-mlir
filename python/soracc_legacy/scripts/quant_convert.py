# type: ignore

import argparse
from safetensors.numpy import load_file, save_file
import torch # type: ignore
import numpy as np

WEIGHT_PATH = '/data1/shared/OpenSora/weight/model.safetensors'
SCALE_PATH = '/data1/shared/OpenSora/scale/ckpt.pth'
DUMP_PATH = '/data1/shared/OpenSora/model_quant.safetensors'

def parse_args():
    parser = argparse.ArgumentParser(description='Convert quantized model to float model')
    parser.add_argument('--weight_path', type=str, default=WEIGHT_PATH, help='path to quantized weight file')
    parser.add_argument('--scale_path', type=str, default=SCALE_PATH, help='path to scale file')
    parser.add_argument('--dump_path', type=str, default=DUMP_PATH, help='path to save float model')

    args = parser.parse_args()
    return args

def convert_scale_to_safetensors(pth_path, safetensors_path):
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        if v[0]['delta'] is None:
            continue
        else:
            new_state_dict[k] = v[0]['delta'].to(torch.float16).numpy().astype(np.float16)
    save_file(new_state_dict, safetensors_path)

def split_qkv_weight(weight_data):
    new_weight_data = {}
    for k, v in weight_data.items():
        if 'qkv' in k:
            qkv_weight = v
            if 'weight' in k:
                qkv_weight = np.split(qkv_weight, 3, axis=-2)
            elif 'bias' in k:
                qkv_weight = np.split(qkv_weight, 3, axis=0)

            new_weight_data[k.replace('qkv', 'q')] = qkv_weight[0]
            new_weight_data[k.replace('qkv', 'k')] = qkv_weight[1]
            new_weight_data[k.replace('qkv', 'v')] = qkv_weight[2]
        else:
            new_weight_data[k] = v
    return new_weight_data

def split_kv_weight(weight_data):
    new_weight_data = {}
    for k, v in weight_data.items():
        if 'kv' in k:
            kv_weight = v
            if 'weight' in k:
                kv_weight = np.split(kv_weight, 2, axis=-2)
            elif 'bias' in k:
                kv_weight = np.split(kv_weight, 2, axis=0)

            new_weight_data[k.replace('kv', 'k')] = kv_weight[0]
            new_weight_data[k.replace('kv', 'v')] = kv_weight[1]
        else:
            new_weight_data[k] = v
    return new_weight_data

def quantize_weight_to_int8(weight_data):
    new_weight_data = {}
    for k, v in weight_data.items():
        if ('linear' in k or 'mlp' in k or 'proj' in k or 'q.' in k or 'k.' in k or 'v.' in k) \
            and 'weight' in k and 'quantizer' not in k and 'x_embedder' not in k:
            # scale = weight_data[k.replace('weight', 'weight_quantizer')]
            scale_new = np.max(np.abs(v.astype(np.float32)), axis=-1, keepdims=True) / 127.0
            new_weight_data[k] = np.clip(np.round(v / scale_new), a_min=-128, a_max=127).astype(np.int8)
            weight_data[k.replace('weight', 'weight_quantizer')] = scale_new.astype(np.float32)
        elif 'weight_quantizer' in k:
            new_weight_data[k] = v.astype(np.float32)
        else:
            new_weight_data[k] = v.astype(np.float16)
    return new_weight_data

def add_cos_sin_table(weight_data: dict, dim: int = 72, max_len: int = 512):
    assert "rope.freqs" in weight_data.keys()
    assert weight_data["rope.freqs"].shape == (dim//2,)
    freq = weight_data["rope.freqs"]
    t = np.arange(0, max_len)

    freqs = np.expand_dims(t, axis=-1) * freq
    freqs = np.repeat(freqs, 2, axis=-1)
    
    cos_table = np.cos(freqs)
    sin_table = np.sin(freqs) * np.array([(-1) ** i for i in range(dim)])

    cos_sin_table = np.concatenate([cos_table, sin_table], axis=-1)
    weight_data["rope.cos_sin_table"] = cos_sin_table.astype(np.float16)

    return weight_data

if __name__ == '__main__':
    args = parse_args()
    weight_path = args.weight_path
    scale_path = args.scale_path

    assert weight_path.endswith('.safetensors')
    assert scale_path.endswith('.pth')

    convert_scale_to_safetensors(scale_path, scale_path.replace('.pth', '.safetensors'))
    scale_path = scale_path.replace('.pth', '.safetensors')

    weight_data = load_file(weight_path)
    scale_data = load_file(scale_path)
    
    merged_data = {**weight_data, **scale_data}
    
    merged_data = add_cos_sin_table(merged_data)

    new_quant_weight_data = quantize_weight_to_int8(merged_data)

    new_quant_weight_data = split_qkv_weight(new_quant_weight_data)
    new_weight_data = split_kv_weight(new_quant_weight_data)

    save_file(new_weight_data, args.dump_path)
    print(f'Quantized model converted to float model, saved to {args.dump_path}')
