import json
import argparse
import inst

parser = argparse.ArgumentParser("sora compiler")
parser.add_argument('--infile', type=str, default='asm.json')
parser.add_argument('--outfile', type=str, default='inst.bin')
args = parser.parse_args()

def main():
    inst_collect = inst.InstCollector()
    with open(args.infile, 'r') as fin, open(args.outfile, 'wb') as fout:
        inst_js = json.load(fin)
        inst_collect.from_json(inst_js)
        inst_bits = inst.insts_to_bin(inst_collect.get_insts())
        fout.write(inst_bits)
        
if __name__ == '__main__':
    main()
