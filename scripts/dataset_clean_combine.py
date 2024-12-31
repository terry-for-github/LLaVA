import os
import json
from PIL import Image
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='解析第一个参数作为JSON文件')
parser.add_argument('json_file', type=str, help='输入的JSON文件')
parser.add_argument('tot', type=int, help='总线程数')
arg = parser.parse_args()
json_path = arg.json_file
tot = arg.tot
tot_list_data_dict = []
original_list_data_dict = json.load(open(json_path, "r"))
for idx in tqdm(range(tot)):
    cur_json_path = json_path.replace(".json", f"_{idx}_{tot}.json")
    with open(cur_json_path, "r") as f:
        list_data_dict = json.load(f)
        tot_list_data_dict.extend(list_data_dict)
    os.remove(cur_json_path)
json.dump(tot_list_data_dict, open(json_path.replace('.json', '_exist.json'), "w"), indent=2, ensure_ascii=False)

print(f"Total {len(original_list_data_dict)} -> {len(tot_list_data_dict)}")
