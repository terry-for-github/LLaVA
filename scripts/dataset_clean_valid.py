import os
import json
from PIL import Image
from tqdm import tqdm
import argparse
import string
from transformers import AutoImageProcessor


parser = argparse.ArgumentParser(description='解析第一个参数作为JSON文件')
parser.add_argument('json_file', type=str, help='输入的JSON文件')
parser.add_argument('idx', type=int, help='第i个线程')
parser.add_argument('tot', type=int, help='总线程数')
arg = parser.parse_args()
json_path = arg.json_file
idx = arg.idx
tot = arg.tot

# 用来过滤一些，用英文提问，但是中文回答的样本
# 这里的逻辑是，如果大部分字符都是英文的（>70%），那就认为是英文的
def is_english_or_numeric_or_punctuation(text):
    # 匹配只包含英文字符、数字、标点符号的字符串
    valid_characters = string.ascii_letters + string.digits + string.punctuation + ' ' + '\n'
    return len([1 for char in text if char not in valid_characters]) / len(text) < 0.3

if idx == 0:
    print('Start loading data...')

tot_list_data_dict = []
if idx == 0:
    print(f"Loading {json_path}")
if "blip_laion_cc_sbu_558k.json" in json_path:
    path_prefix = "./playground/pretrain/images/"
elif "DCI_8K.json" in json_path:
    path_prefix = "./playground/image_caption/DCI/"
elif "DF_1M" in json_path or "DF_100K" in json_path:
    path_prefix = "./playground/image_caption/DenseFusion/"
elif "DOCCI_15K.json" in json_path:
    path_prefix = "./playground/image_caption/DOCCI/"
elif "MMInstruct-18K.json" in json_path:
    path_prefix = "./playground/image_caption/MMInstruct/"
elif "ShareGPT4V_102K.json" in json_path:
    path_prefix = "./playground/image_caption/ShareGPT4V/"
elif "GBC-10M" in json_path:
    path_prefix = "./playground/image_caption/"
else:
    raise ValueError(f"Unknown dataset: {json_path}")

list_data_dict = json.load(open(json_path, "r"))
if idx == 0:
    print(f"Processing {json_path}")
start_idx = idx * len(list_data_dict) // tot
end_idx = (idx + 1) * len(list_data_dict) // tot
list_data_dict = list_data_dict[start_idx:end_idx]
clean_list_data_dict = []
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

for json_idx, data_dict in enumerate(tqdm(list_data_dict)):
    if 'image' not in data_dict:
        print('No image', json_idx+start_idx)
        continue
    image_path = os.path.join(path_prefix, data_dict['image'])
    if not os.path.exists(image_path):
        # print('Image not exists', json_idx+start_idx, image_path)
        continue
    try:
        img = Image.open(image_path).convert("RGB")
        processor(img, return_tensors="pt")
    except Exception as e:
        print('Error in image', json_idx+start_idx, e)
        continue
    flag = False
    for message in data_dict['conversations']:
        from_ = message['from']
        value = message['value']
        if from_ == 'human':
            if '<image>' not in value:
                print('Image not in value', json_idx+start_idx)
                flag = True
                break
        elif 'gpt' not in from_:
            print('gpt not in from_', json_idx+start_idx)
            flag = True
            break
        message['value'] = value.encode('utf-8', 'ignore').decode('utf-8')
        if len(message['value']) == 0:
            print('Empty value', json_idx+start_idx)
            flag = True
            break
    if flag:
        continue
    clean_list_data_dict.append(data_dict)
json.dump(clean_list_data_dict, open(f"{json_path[:-5]}_{idx}_{tot}.json", "w"), indent=2, ensure_ascii=False)
