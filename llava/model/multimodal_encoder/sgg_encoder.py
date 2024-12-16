import os
from collections import OrderedDict
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import torch
import shutil


def draw_(image_path, boxes, labels, rel_labels, rel_pairs, output_path):
    # 打开图片
    img = Image.open(image_path)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 左边绘制带有不同颜色的 bounding box 图
    ax1 = axes[0]
    ax1.imshow(img)
    ax1.axis('off')

    # 随机生成颜色
    num_boxes = len(boxes)
    colors = [plt.cm.tab20(i) for i in range(num_boxes)]  # 使用 tab20 调色板
    width, height = img.size
    whole_image_rect = plt.Rectangle((0, 0), width, height, linewidth=3, edgecolor='blue', facecolor='none', alpha=0.5)
    ax1.add_patch(whole_image_rect)
    # 遍历所有的bounding box并画出来
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # assert 0 <= x1 <= x2 <= 1, f'x1: {x1}, x2: {x2}'
        # assert 0 <= y1 <= y2 <= 1, f'y1: {y1}, y2: {y2}'
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
        label = labels[i]
        color = colors[i]

        # 画出bounding box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        # 添加标签
        ax1.text(x1, y1, label, color=color, fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.5))

    ax2 = axes[1]
    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(i, label=label)

    # 添加边和关系
    for i, (subj, obj) in enumerate(rel_pairs):
        G.add_edge(subj.item(), obj.item(), label=rel_labels[i])

    # 绘制图
    pos = nx.spring_layout(G, k=2)  # 使用 spring layout 布局
    for i, (node, color) in enumerate(zip(G.nodes(), colors)):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[color], ax=ax2)
    nx.draw_networkx_edges(G, pos, ax=ax2)

    nx.draw_networkx_labels(G, pos, labels={i: labels[i] for i in G.nodes()}, font_size=10, ax=ax2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['label'] for u, v in G.edges()}, font_size=10, ax=ax2, label_pos=0.25, bbox=dict(facecolor='white', alpha=0.5))

    # 保存图片
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def time_recorder(func):
    def wrapper(*args, **kwargs):
        torch.distributed.barrier()
        import time
        start = time.time()
        res = func(*args, **kwargs)
        print(f'{func.__name__} cost: {time.time() - start}')
        torch.distributed.barrier()
        return res
    return wrapper


class PostProcessGraph(nn.Module):
    def __init__(self, num_boxes=32, num_relations=32, box_threshold=0.2, rel_threshold=0.03):
        super(PostProcessGraph, self).__init__()
        self.num_boxes = num_boxes
        self.num_relations = num_relations
        self.box_threshold = box_threshold
        self.rel_threshold = rel_threshold
    trans_classes = {0: 'airplane', 1: 'animal', 2: 'arm', 3: 'bag', 4: 'banana', 5: 'basket', 6: 'beach', 7: 'bear', 8: 'bed', 9: 'bench', 10: 'bike', 11: 'bird', 12: 'board', 13: 'boat', 14: 'book', 15: 'boot', 16: 'bottle', 17: 'bowl', 18: 'box', 19: 'boy', 20: 'branch', 21: 'building', 22: 'bus', 23: 'cabinet', 24: 'cap', 25: 'car', 26: 'cat', 27: 'chair', 28: 'child', 29: 'clock', 30: 'coat', 31: 'counter', 32: 'cow', 33: 'cup', 34: 'curtain', 35: 'desk', 36: 'dog', 37: 'door', 38: 'drawer', 39: 'ear', 40: 'elephant', 41: 'engine', 42: 'eye', 43: 'face', 44: 'fence', 45: 'finger', 46: 'flag', 47: 'flower', 48: 'food', 49: 'fork', 50: 'fruit', 51: 'giraffe', 52: 'girl', 53: 'glass', 54: 'glove', 55: 'guy', 56: 'hair', 57: 'hand', 58: 'handle', 59: 'hat', 60: 'head', 61: 'helmet', 62: 'hill', 63: 'horse', 64: 'house', 65: 'jacket', 66: 'jean', 67: 'kid', 68: 'kite', 69: 'lady', 70: 'lamp', 71: 'laptop', 72: 'leaf', 73: 'leg', 74: 'letter', 75: 'light', 76: 'logo', 77: 'man', 78: 'men', 79: 'motorcycle', 80: 'mountain', 81: 'mouth', 82: 'neck', 83: 'nose', 84: 'number', 85: 'orange', 86: 'pant', 87: 'paper', 88: 'paw', 89: 'people', 90: 'person', 91: 'phone', 92: 'pillow', 93: 'pizza', 94: 'plane', 95: 'plant', 96: 'plate', 97: 'player', 98: 'pole', 99: 'post', 100: 'pot', 101: 'racket', 102: 'railing', 103: 'rock', 104: 'roof', 105: 'room', 106: 'screen', 107: 'seat', 108: 'sheep', 109: 'shelf', 110: 'shirt', 111: 'shoe', 112: 'short', 113: 'sidewalk', 114: 'sign', 115: 'sink', 116: 'skateboard', 117: 'ski', 118: 'skier', 119: 'sneaker', 120: 'snow', 121: 'sock', 122: 'stand', 123: 'street', 124: 'surfboard', 125: 'table', 126: 'tail', 127: 'tie', 128: 'tile', 129: 'tire', 130: 'toilet', 131: 'towel', 132: 'tower', 133: 'track', 134: 'train', 135: 'tree', 136: 'truck', 137: 'trunk', 138: 'umbrella', 139: 'vase', 140: 'vegetable', 141: 'vehicle', 142: 'wave', 143: 'wheel', 144: 'window', 145: 'windshield', 146: 'wing', 147: 'wire', 148: 'woman', 149: 'zebra'}
    trans_rel = {0: 'above', 1: 'across', 2: 'against', 3: 'along', 4: 'and', 5: 'at', 6: 'attached to', 7: 'behind', 8: 'belonging to', 9: 'between', 10: 'carrying', 11: 'covered in', 12: 'covering', 13: 'eating', 14: 'flying in', 15: 'for', 16: 'from', 17: 'growing on', 18: 'hanging from', 19: 'has', 20: 'holding', 21: 'in', 22: 'in front of', 23: 'laying on', 24: 'looking at', 25: 'lying on', 26: 'made of', 27: 'mounted on', 28: 'near', 29: 'of', 30: 'on', 31: 'on back of', 32: 'over', 33: 'painted on', 34: 'parked on', 35: 'part of', 36: 'playing', 37: 'riding', 38: 'says', 39: 'sitting on', 40: 'standing on', 41: 'to', 42: 'under', 43: 'using', 44: 'walking in', 45: 'walking on', 46: 'watching', 47: 'wearing', 48: 'wears', 49: 'with'}

    def get_scale(self, image_size):
        width, height = image_size
        
    def forward(self, outputs, image_sizes, image_paths):
        """
        Post-process the model's outputs to build the final graph representation

        Parameters:
            outputs: Raw outputs of the model (predictions for boxes, classes, and relations)
            clip_feature: Visual feature map of shape (batch_size, 576, 4096)
            image_sizes: Sizes of images in the batch (batch_size, 2)
        
        Returns:
            clip_feature: Processed clip feature
            boxes: Final boxes after filtering
            relations: Final relations after filtering
            adj_matrix: Adjacency matrix representing the graph
        """
        result_list = []
        # clip_feature: torch.bfloat16 (batch_size, 576, 4096)
        # print_output(result_list)
        # result_list: list with length `batch_size`
        # result_list[i]['graph']: dict where:{
        #   'node_id': torch.int64 (100,)
        #   'pred_boxes': torch.float32 (100, 4)
        #   'pred_boxes_score': torch.float32 (100,)
        #   'pred_boxes_class': torch.int64 (100,)
        #   'all_node_pairs': torch.int64 (9900, 2)
        #   'all_relation': torch.bfloat16 (9900, 51)
        #   'rel_classes': torch.int64 (9900,)
        #   'rel_scores': torch.float32 (9900,)
        #   'rln_features': torch.bfloat16 (9900, 256)
        # }
        # 1. delete the boxes with score < 0.2
        # 2. also delete the relation that contains the deleted boxes
        # 3. delete the relation with score < 0.2
        # 4. keep at most 32 boxes and 32 relations
        # 5. use the remaining relation to build a graph adj matrix
        # (
        # a square matrix where side size = 576 + 32 + 32
        # if the num_boxes and num_relations less than 32, padding to 32 and keeping it empty
        # if node i and node j have a relation k, then adj[i, k] = 1, adj[k, j] = 1
        # if a node i's boxes cover the clip_feature[a:b, c:d], then adj[i, 24*x+y] = 1 adj[24*x+y, i] = 1 for x in [a, b), y in [c, d)
        # )
        # 6. return the clip_feature, the boxes, the relations, and the adj matrix
        # results = [result['graph']['rln_features'][:self.num_patches, :] for result in result_list]
        for i in range(len(outputs)):
            result = {}

            # Extract outputs for this particular image in the batch
            graph = outputs[i]['graph']
            pred_boxes = graph['pred_boxes']  # shape (100, 4)
            pred_scores = graph['pred_boxes_score']  # shape (100,)
            pred_classes = graph['pred_boxes_class']  # shape (100,)
            all_rel_pairs = graph['all_node_pairs']  # shape (9900, 2)
            all_rel_scores = graph['rel_scores']  # shape (9900,)
            all_rel_classes = graph['rel_classes']  # shape (9900,)
            rln_features = graph['rln_features']  # shape (9900, 256)
            dtype = rln_features.dtype

            # 1. Delete the boxes with score < score_threshold
            keep_boxes = pred_scores > self.box_threshold
            pred_boxes = pred_boxes[keep_boxes]
            pred_scores = pred_scores[keep_boxes]
            pred_classes = pred_classes[keep_boxes]
            keep_boxes[self.num_boxes:] = False

            # 2. Delete relations with score < score_threshold
            keep_relations = all_rel_scores > self.rel_threshold
            all_rel_pairs = all_rel_pairs[keep_relations]
            all_rel_scores = all_rel_scores[keep_relations]
            all_rel_classes = all_rel_classes[keep_relations]
            # rln_features = rln_features[keep_relations]

            # 3. Delete relations that involve deleted boxes (after filtering boxes)
            src_keep = keep_boxes[all_rel_pairs[:, 0]]  # 获取 src 是否保留
            dst_keep = keep_boxes[all_rel_pairs[:, 1]]  # 获取 dst 是否保留
            keep_relations = src_keep & dst_keep  # 同时满足 src 和 dst 为 True

            # 使用布尔索引过滤 all_node_pairs
            all_rel_pairs = all_rel_pairs[keep_relations]
            all_rel_scores = all_rel_scores[keep_relations]
            all_rel_classes = all_rel_classes[keep_relations]
            # rln_features = rln_features[keep_relations]
            # 4. Keep at most num_boxes boxes and num_relations relations
            num_boxes = min(self.num_boxes, len(pred_boxes))
            num_relations = min(self.num_relations, len(all_rel_pairs))

            pred_boxes = pred_boxes[:num_boxes]
            pred_scores = pred_scores[:num_boxes]
            pred_classes = pred_classes[:num_boxes]

            all_rel_pairs = all_rel_pairs[:num_relations]
            all_rel_scores = all_rel_scores[:num_relations]
            all_rel_classes = all_rel_classes[:num_relations]
            # rln_features = rln_features[:num_relations]

            # 5. Build the adjacency matrix
            num_clip_nodes = 576
            num_box_nodes = self.num_boxes
            num_relation_nodes = self.num_relations
            num_text_nodes = self.num_boxes + self.num_relations
            total_nodes = num_clip_nodes + num_box_nodes + num_relation_nodes

            adj_matrix = torch.zeros((total_nodes, total_nodes), dtype=dtype, device=pred_boxes.device)

            # Mapping for node types
            clip_node_start = 0
            box_node_start = num_clip_nodes
            rel_node_start = num_clip_nodes + num_box_nodes

            # Add box-clip feature connections (based on coverage)
            concat_image_sizes = torch.concatenate((image_sizes[i], image_sizes[i]), dim=-1)
            float_boxes = pred_boxes / concat_image_sizes
            # output_path = './playground/debug/' + os.path.basename(image_paths[i])
            # os.makedirs('./playground/debug', exist_ok=True)
            # box_labels = [str(idx)+ '-' + self.trans_classes[label.item()-1] + '-' + str(pred_scores[idx].item())[:4] for idx, label in enumerate(pred_classes)]
            # rel_labels = [self.trans_rel[rel.item()-1]for idx, rel in enumerate(all_rel_classes)]
            # draw_(image_paths[i], float_boxes.cpu(), box_labels, rel_labels, all_node_pairs, output_path)
            
            pred_boxes = (torch.clamp(float_boxes, 0., 0.999) * 24).int()
            box_x1, box_y1, box_x2, box_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

            x_range = torch.arange(24, device=pred_boxes.device)  # 假设最大范围是 24
            y_range = torch.arange(24, device=pred_boxes.device)
            x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')

            # 创建一个 mask，找出哪些点在 box 内
            x_mask = (x_grid.unsqueeze(0) >= box_x1.unsqueeze(1).unsqueeze(2)) & (x_grid.unsqueeze(0) <= box_x2.unsqueeze(1).unsqueeze(2))
            y_mask = (y_grid.unsqueeze(0) >= box_y1.unsqueeze(1).unsqueeze(2)) & (y_grid.unsqueeze(0) <= box_y2.unsqueeze(1).unsqueeze(2))
            mask = x_mask & y_mask  # 组合 x 和 y 方向的条件
            x_grid_expanded = x_grid.unsqueeze(0).expand(mask.shape)
            y_grid_expanded = y_grid.unsqueeze(0).expand(mask.shape)
            # 获取在 mask 中为 True 的坐标
            clip_indices = (x_grid_expanded[mask] * 24 + y_grid_expanded[mask]).reshape(-1)
            box_indices = torch.arange(len(pred_boxes), device=pred_boxes.device).repeat_interleave(mask.sum(dim=(1, 2)))

            # 计算 clip 和 box 对应的节点索引
            clip_nodes = rel_node_start + clip_indices
            box_nodes = box_node_start + box_indices

            adj_matrix[box_nodes, clip_nodes] = 1.
            adj_matrix[clip_nodes, box_nodes] = 1.

            # 计算 src, dst, rel 的索引位置
            src_nodes = all_rel_pairs[:, 0]
            dst_nodes = all_rel_pairs[:, 1]
            rel_nodes = rel_node_start + torch.arange(num_relations)

            adj_matrix[src_nodes, rel_nodes] = 1.
            adj_matrix[rel_nodes, dst_nodes] = 1.

            result['nodes'] = {
                'coordinates': pred_boxes,
                'scores': pred_scores,
                'classes': pred_classes
            }
            result['relations'] = {
                'node_pairs': all_rel_pairs,
                'relation_scores': all_rel_scores,
                'relation_classes': all_rel_classes
            }
            result['adj_matrix'] = adj_matrix

            result_list.append(result)

        return result_list


class SceneGraphVisionTower(nn.Module):
    cfg_path = '/userhome/llava_code/ovsgtr_ours/config/GroundingDINO_SwinB_full.py'
    ckpt_path = '/userhome/OvSGTR/vg-swinb-full.pth'
    VG150_OBJ_CATEGORIES = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear',
                            'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy',
                            'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat',
                            'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant',
                            'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit',
                            'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet',
                            'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf',
                            'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck',
                            'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow',
                            'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing',
                            'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short',
                            'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand',
                            'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower',
                            'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave',
                            'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']


    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.image_id = 0
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 调参
        self.num_boxes = 32
        self.num_relations = 32
        self.box_threshold = 0.2
        self.rel_threshold = 0.03
        self.llm_hidden_size = 3072
        self.sgma = PostProcessGraph(
            num_boxes=self.num_boxes,
            num_relations=self.num_relations,
            box_threshold=self.box_threshold,
            rel_threshold=self.rel_threshold
        )

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SLConfig.fromfile(self.cfg_path)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SceneGraphImageProcessor()

        self.args = SLConfig.fromfile(self.cfg_path)
        self.args.device = "cuda"
        self.vision_tower, criterion, self.postprocessors = build_model(self.args)
        self.vision_tower.to(device='cuda', dtype=torch.bfloat16)
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        load_res = self.vision_tower.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        rln_proj = getattr(self.vision_tower, "rln_proj", None)
        rln_classifier = getattr(self.vision_tower, "rln_classifier", None)
        rln_freq_bias = getattr(self.vision_tower, "rln_freq_bias", None)
        self.postprocessors['bbox'].rln_proj = rln_proj
        self.postprocessors['bbox'].rln_classifier = rln_classifier
        self.postprocessors['bbox'].rln_freq_bias = rln_freq_bias
        name2classes = OrderedDict({name: idx for idx, name in enumerate(self.VG150_OBJ_CATEGORIES) if name != '__background__'})
        self.postprocessors['bbox'].name2classes = name2classes
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # 原来是B x 9900 x 256，现在只保留前20
        return image_forward_outs[:self.num_patches]

    caption = 'airplane. animal. arm. bag. banana. basket. beach. bear. bed. bench. bike. bird. board. boat. book. boot. bottle. bowl. box. boy. branch. building. bus. cabinet. cap. car. cat. chair. child. clock. coat. counter. cow. cup. curtain. desk. dog. door. drawer. ear. elephant. engine. eye. face. fence. finger. flag. flower. food. fork. fruit. giraffe. girl. glass. glove. guy. hair. hand. handle. hat. head. helmet. hill. horse. house. jacket. jean. kid. kite. lady. lamp. laptop. leaf. leg. letter. light. logo. man. men. motorcycle. mountain. mouth. neck. nose. number. orange. pant. paper. paw. people. person. phone. pillow. pizza. plane. plant. plate. player. pole. post. pot. racket. railing. rock. roof. room. screen. seat. sheep. shelf. shirt. shoe. short. sidewalk. sign. sink. skateboard. ski. skier. sneaker. snow. sock. stand. street. surfboard. table. tail. tie. tile. tire. toilet. towel. tower. track. train. tree. truck. trunk. umbrella. vase. vegetable. vehicle. wave. wheel. window. windshield. wing. wire. woman. zebra.'
    rel_caption = 'above. across. against. along. and. at. attached to. behind. belonging to. between. carrying. covered in. covering. eating. flying in. for. from. growing on. hanging from. has. holding. in. in front of. laying on. looking at. lying on. made of. mounted on. near. of. on. on back of. over. painted on. parked on. part of. playing. riding. says. sitting on. standing on. to. under. using. walking in. walking on. watching. wearing. wears. with.'
    node_list = ['none', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
    edge_list = ['none', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
    node_features = None
    edge_features = None

    @torch.no_grad()
    def forward(self, images, image_sizes, image_features, image_paths, embed_tokens):
        # return torch.zeros(images.size(0), self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)
        if self.node_features is None:
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
            with torch.no_grad():
                self.node_features = torch.concat(
                    [embed_tokens(torch.tensor(node, device=self.device)).mean(dim=0, keepdim=True)
                     for node in tokenizer.batch_encode_plus(self.node_list, add_special_tokens=False).input_ids]
                )
        if self.edge_features is None:
            self.edge_features = torch.concat(
                [embed_tokens(torch.tensor(rel, device=self.device)).mean(dim=0, keepdim=True)
                    for rel in tokenizer.batch_encode_plus(self.edge_list, add_special_tokens=False).input_ids]
            )
        batch_size = len(images)

        def print_output(output, key=None):
            if isinstance(output, torch.Tensor):
                print(key, output.shape, output.dtype)
            elif isinstance(output, list):
                print_output(output[0], str(key)+f'(list{len(output)})')
            elif isinstance(output, dict):
                for k, v in output.items():
                    print_output(v, str(key)+'.'+k)
            else:
                print(key, output)
        outputs = self.vision_tower(
            images,
            captions=[self.caption for _ in range(batch_size)],
            rel_captions=[self.rel_caption for _ in range(batch_size)]
        )
        result_list = self.postprocessors['bbox'](outputs, image_sizes)
        result_list = self.sgma(outputs=result_list, image_sizes=image_sizes, image_paths=image_paths)
        adj_matrix = torch.stack([result['adj_matrix'] for result in result_list], dim=0)
        node_embeddings = torch.zeros((batch_size, 576 + self.num_boxes + self.num_relations, self.llm_hidden_size), device=self.device, dtype=self.dtype)
        node_embeddings[:, :576, :] = image_features
        for idx, result in enumerate(result_list):
            node_classes = result['nodes']['classes']
            node_embeddings[idx, 576:576+self.num_boxes, :] = self.node_features[node_classes]
            rel_classes = result['relations']['relation_classes']
            node_embeddings[idx, 576+self.num_boxes:, :] = self.edge_features[rel_classes]
        # result['adj_matrix'] = adj_matrix
        # result['nodes'] = {
        #     'coordinates': pred_boxes,
        #     'scores': pred_scores,
        #     'classes': pred_classes
        # }
        # result['relations'] = {
        #     'node_pairs': all_node_pairs,
        #     'relation_scores': all_relation_scores,
        #     'relation_classes': all_rel_classes
        # }
        # result_list[i]['adj_matrix']: torch.bfloat16 (576+20+20, 576+20+20)
        # result_list[i]['nodes'] = {
        #     'coordinates': pred_boxes, torch.bfloat16 (20, 4)
        #     'scores': pred_scores, torch.bfloat16 (20,)
        #     'classes': pred_classes, torch.int64 (20,)
        # }
        # result_list[i]['relations'] = {
        #     'node_pairs': all_node_pairs, torch.int64 (20, 2)
        #     'relation_scores': all_relation_scores, torch.bfloat16 (20,)
        #     'relation_classes': all_rel_classes, torch.int64 (20, )
        # }
        # print_output(result_list)
        return adj_matrix, node_embeddings
        rel_scores = [result['graph']['rel_scores'] for result in result_list]
        # rel_scores = torch.stack(rel_scores, dim=0)
        bboxes = [result['graph']['pred_boxes'] for result in result_list]
        all_node_pairs = [result['graph']['all_node_pairs'] for result in result_list]
        rel_classes = [result['graph']['rel_classes'] for result in result_list]
        results = {}
        keep_rln_features = []
        keep_sub_bboxes = []
        keep_obj_bboxes = []
        keep_pairs = []
        keep_rels = []
        keep_boxes = []
        for i, (rel_score, rln_feature, bbox, all_node_pair, rel_class) in enumerate(zip(rel_scores, rln_features, bboxes, all_node_pairs, rel_classes)):
            index = torch.nonzero(rel_score > 0.08).squeeze(-1)
            if index.shape[0] == 0:
                keep_rln_features.append([])
                keep_pairs.append([])
                keep_rels.append([])
                keep_sub_bboxes.append([])
                keep_obj_bboxes.append([])
            else:
                keep_rln_features.append(rln_feature[index])
                keep_pairs.append(all_node_pair[index])
                keep_sub_bboxes.append(bbox[all_node_pair[:, 0]][index])
                keep_obj_bboxes.append(bbox[all_node_pair[:, 1]][index])
                keep_rels.append(rel_class[index])
                sub_ids = all_node_pair[:, 0][index]
                obj_ids = all_node_pair[:, 1][index]
              
                all_box_indexes = torch.cat((sub_ids, obj_ids))
                unique_tensor = torch.unique(all_box_indexes)
                # print(sub_ids, obj_ids, unique_tensor)
                mask = torch.zeros((unique_tensor.shape[0], unique_tensor.shape[0]))
                for sub_id, obj_id in zip(sub_ids, obj_ids):
                    sub_indice = torch.where(unique_tensor == sub_id)[0]
                    obj_indice = torch.where(unique_tensor == obj_id)[0]
                    mask[sub_indice, obj_indice] = 1
                    # print(sub_indice, obj_indice)
                    
        results['keep_sub_bboxes'] = keep_sub_bboxes
        results['keep_obj_bboxes'] = keep_obj_bboxes
        results['keep_rels'] = keep_rels
        results['keep_rln_features'] = keep_rln_features ### relation视觉特征
        results['keep_pairs'] = keep_pairs
        
        # if os.environ['RANK'] == '0':
        #     for i, result in enumerate(result_list): 
        #         boxes = result['graph']['pred_boxes'].cpu().numpy()
        #         # scores = result['graph']['pred_boxes_score'].cpu().numpy()
        #         labels = result['graph']['pred_boxes_class'].cpu().numpy()
        #         # draw_boxes(image_pil, scores[:5], boxes[:5], labels[:5])
        #         all_node_pairs = result['graph']['all_node_pairs'].cpu().numpy()
        #         rel_classes = result['graph']['rel_classes'].cpu().numpy()
        #         # rln_features = result['graph']['rln_features'].cpu()
        #          np.savez('results/save_{}.npz'.format(i + 1), boxes=boxes,labels=labels, all_node_pairs=all_node_pairs, rel_classes=rel_classes)
        # self.image_id += 1
        
        return results


    

    def encode_graph_entities_embedding(sub_embeddings, obj_embeddings, rel_embeddings):
        
        pass

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.num_patches, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.args

    @property
    def hidden_size(self):
        return 256

    @property
    def num_patches_per_side(self):
        raise NotImplementedError('undefined')

    @property
    def num_patches(self):
        return 20


class SceneGraphImageProcessor:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, pil_image, return_tensors='pt'):
        assert return_tensors == 'pt'
        image_tensor, _ = self.transform(pil_image, None)
        return dict(pixel_values=image_tensor.unsqueeze(0))


