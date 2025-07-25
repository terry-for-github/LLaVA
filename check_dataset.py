import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor


def expand2square(pil_img: Image.Image, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result

if __name__ == '__main__':
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    assert isinstance(image_processor, CLIPImageProcessor)
    data_root = 'playground/onevision'
    dataset_list = os.listdir(data_root)
    cnt = 0
    for checking_index in tqdm(range(len(dataset_list))):
        # checking_index = 35
        dataset = dataset_list[checking_index]
        if dataset not in ['llava_next_raw_format', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mathqa']:
            continue
        print(dataset)
        # print(checking_index, dataset)
        data_root = 'playground/onevision'
        dataset_root = os.path.join(data_root, dataset)
        if dataset in ['cambrian(filtered)', 'ureader_qa', 'ureader_kg', 'llava_next_raw_format']:
            json_file = os.path.join(dataset_root, dataset+'_processed.json')
            data_root = dataset_root
        else:
            json_file = os.path.join(dataset_root, dataset+'_anno.json')
            wrong_file = os.path.join(dataset_root, dataset+'_wrong.json')
            list_wrong = json.load(open(wrong_file, 'r'))
            assert isinstance(list_wrong, list), list_wrong
            assert len(list_wrong) == 0, (dataset, len(list_wrong))
        list_data_dict = json.load(open(json_file, 'r'))
        # factor = 1.
        # if dataset in ['sroie', 'hme100k', 'tallyqa(cauldron,llava_format)']:
        #     factor = 0.1
        #     print(dataset, len(list_data_dict), factor)
        # elif dataset in ['k12_printing', 'clevr(cauldron,llava_format)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)']:
        #     factor = 0.01
        #     print(dataset, len(list_data_dict), factor)
        # elif dataset in ['scienceqa(nona_context)', 'FigureQA(MathV360K)', 'IconQA(MathV360K)', 'PMC-VQA(MathV360K)', 'raven(cauldron)', 'iconqa(cauldron,llava_format)', 'tqa(cauldron,llava_format)']:
        #     factor = 0.05
        #     print(dataset, len(list_data_dict), factor)
        # cnt += int(len(list_data_dict) * factor)
        for data_dict in tqdm(list_data_dict):
            assert isinstance(data_dict, dict)
            if dataset not in ['magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mathqa']:
                assert 'image' in data_dict and 'conversations' in data_dict
                assert os.path.exists(os.path.join(data_root, data_dict['image']))
            if dataset == 'lrv_normal(filtered)' and data_dict['id'] in [1475, 4675, 7527]:
                continue
            assert isinstance(data_dict['conversations'], list) and len(data_dict['conversations']) % 2 == 0, data_dict
            for i, message in enumerate(data_dict['conversations']):
                assert isinstance(message, dict) and 'from' in message and 'value' in message
                assert message['from'] == ('human' if i%2==0 else 'gpt'), data_dict
                assert isinstance(message['value'], str)
                if dataset in ['magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mathqa']:
                    continue
                if dataset in ['websight(cauldron)', 'multihiertt(cauldron)', 'geomverse(cauldron)', 'dvqa(cauldron,llava_format)',
                            'visual7w(cauldron,llava_format)', 'iconqa(cauldron,llava_format)', 'aokvqa(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'screen2words(cauldron)',
                            'figureqa(cauldron,llava_format)', 'vistext(cauldron)', 'ai2d(cauldron,llava_format)', 'robut_wikisql(cauldron)', 'tabmwp(cauldron)', 'robut_wtq(cauldron,llava_format)',
                            'iam(cauldron)', 'vqarad(cauldron,llava_format)', 'intergps(cauldron,llava_format)', 'hateful_memes(cauldron,llava_format)', 'diagram_image_to_text(cauldron)',
                            'chart2text(cauldron)', 'clevr(cauldron,llava_format)', 'raven(cauldron)', 'visualmrc(cauldron)', 'vsr(cauldron,llava_format)', 'infographic_vqa_llava_format',
                            'rendered_text(cauldron)', 'scienceqa(cauldron,llava_format)', 'mapqa(cauldron,llava_format)', 'tqa(cauldron,llava_format)', 'st_vqa(cauldron,llava_format)',
                            'robut_sqa(cauldron)', 'tallyqa(cauldron,llava_format)', 'chartqa(cauldron,llava_format)']:
                    assert not '<image>' in message['value'], data_dict
                elif dataset in ['lrv_normal(filtered)']:
                    if i != 0:
                        assert '<image>' not in message['value'], data_dict
                elif dataset in ['llavar_gpt4_20k']:
                    if i == 0:
                        assert message['value'].startswith('<image>\n') or message['value'].endswith('\n<image>'), (dataset, data_dict)
                    else:
                        assert not '<image>' in message['value'], message['value']
                elif dataset in ['cambrian(filtered)']:
                    if i == 0:
                        assert '<image>' in message['value'], data_dict
                    else:
                        assert not '<image>' in message['value'], data_dict
                else:
                    if i == 0:
                        assert message['value'].startswith('<image>\n'), (dataset, data_dict)
                    else:
                        assert not '<image>' in message['value'], message['value']
    # for i in range(10):
    #     idx = random.randint(0, len(list_data_dict)-1)
    #     data_dict = list_data_dict[idx]
    #     print(dataset, data_dict['id'], '/', len(list_data_dict))
    #     origin_image = Image.open(data_root + '/' + data_dict['image']).convert('RGB')
    #     print('origin', origin_image.size)
    #     origin_image.save('origin.jpg')

    #     square_image = expand2square(origin_image, tuple(int(x*255) for x in image_processor.image_mean))
    #     square_image = image_processor.preprocess(square_image, do_normalize=False, data_format="channels_last", return_tensors="np")['pixel_values'][0]
    #     square_image = Image.fromarray((255*square_image).astype(np.uint8), mode='RGB')
    #     square_image.save('square.jpg')

    #     clip_image = image_processor.preprocess(origin_image, do_normalize=False, data_format="channels_last", return_tensors="np")['pixel_values'][0]
    #     clip_image = Image.fromarray((255*clip_image).astype(np.uint8), mode='RGB')
    #     # image = expand2square(image, (int(0.48145466*255), int(0.4578275*255), int(0.40821073*255)))
    #     clip_image.save('clip.jpg')

    #     for message in data_dict['conversations']:
    #         print(message['from'], ':', message['value'])
    #     input()


# {%- if tools %}
#     {{- '<|im_start|>system\\n' }}
#     {%- if messages[0]['role'] == 'system' %}
#     {{- messages[0]['content'] }}
#     {%- else %}
#     {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
#     {%- endif %}
#     {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}
#     {%- for tool in tools %}
#         {{- \"\\n\" }}
#         {{- tool | tojson }}
#     {%- endfor %}
#     {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}
# {%- else %}
#     {%- if messages[0]['role'] == 'system' %}
#         {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
#     {%- else %}
#         {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
#     {%- endif %}
# {%- endif %}
# {%- for message in messages %}
#     {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}
#         {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
#     {%- elif message.role == \"assistant\" %}
#         {{- '<|im_start|>' + message.role }}
#         {%- if message.content %}
#             {{- '\\n' + message.content }}
#         {%- endif %}
#         {%- for tool_call in message.tool_calls %}
#             {%- if tool_call.function is defined %}
#                 {%- set tool_call = tool_call.function %}
#             {%- endif %}
#             {{- '\\n<tool_call>\\n{\"name\": \"' }}
#             {{- tool_call.name }}
#             {{- '\", \"arguments\": ' }}
#             {{- tool_call.arguments | tojson }}
#             {{- '}\\n</tool_call>' }}
#         {%- endfor %}
#         {{- '<|im_end|>\\n' }}
#     {%- elif message.role == \"tool\" %}
#         {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}
#             {{- '<|im_start|>user' }}
#         {%- endif %}
#         {{- '\\n<tool_response>\\n' }}
#         {{- message.content }}
#         {{- '\\n</tool_response>' }}
#         {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}
#             {{- '<|im_end|>\\n' }}
#         {%- endif %}
#     {%- endif %}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     {{- '<|im_start|>assistant\\n' }}
# {%- endif %}
