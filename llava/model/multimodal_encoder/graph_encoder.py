from torch.nn import functional as F
import clip
import cv2
import torch
import os
import torch
import json
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from PIL import Image
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors, rel_vectors

SG_HOME = '/userhome/sg_encoder/' # /home/ocl/
SG_MODEL = 'sgdet_trans_baseline/' # trans_baseline/
SG_YAML = 'e2e_relation_X_101_32_8_FPN_1x.yaml' # 'e2e_merge_relation_X_101_32_8_FPN_1x.yaml'

def hex_to_bgr(hex_color):
    # 去掉颜色代码中的'#'号
    hex_color = hex_color.lstrip('#')

    # 解析十六进制颜色值
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)

    # 构建BGR颜色
    bgr_color = (blue, green, red)

    return bgr_color


def draw_bboxes(image, box):

    x, y, x1, y1 = map(int, box)
    cv2.rectangle(image, (x, y), (x1, y1), hex_to_bgr('#8ECFC9'), thickness=2)  # color='#8ECFC9'
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.mlp = nn.Linear(width, 4096)

    def forward(self, x: torch.Tensor):
        return self.mlp(self.resblocks(x))


class GraphProcessor(object):
    def __init__(self):
        self.sg_home = SG_HOME + SG_MODEL
        self.sgg_config_path = SG_HOME + 'pure_sgg_without_csrc/configs/' + SG_YAML
        cfg.merge_from_file(self.sgg_config_path)
        self.transforms = build_transforms(cfg, False)
        self.crop_size = {"height": 600, "width": 600}
        self.img_mean = np.array([102.9801 / 225, 115.9465 / 255, 122.7717 / 255])
        self.iter = 1
    def preprocess(self, img, return_tensors=''):
        target = torch.LongTensor([-1])
        # img.save('/home/pcl/sgg_graph_encoder/demo_{}.png'.format(self.iter))
        self.iter += 1
        img, target = self.transforms(img, target)
        return img


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        return super().forward(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SGAdapter(nn.Module): ## projector = SGAdapter() MLP()
    def __init__(self):
        super().__init__()
        self.sg_home = SG_HOME + SG_MODEL
        self.sgg_config_path = SG_HOME + 'pure_sgg_without_csrc/configs/' + SG_YAML
        self.sgg_model_dir = self.sg_home
        self.sgg_model_path = self.sg_home + 'model_final.pth'
        cache_file = torch.load(self.sg_home + 'VG_stanford_filtered_with_attribute_train_statistics.cache')
        self.rel_classes = cache_file['rel_classes']
        self.obj_classes = cache_file['obj_classes']
        self.hidden_size = 4096
        self.GLOVE_DIR = SG_HOME + 'glove'
        self.embed_dim = 200
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(self.rel_classes, wv_dir=self.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes), self.embed_dim)
        self.rel_embed = nn.Embedding(len(self.rel_classes), self.embed_dim)
        self.projector = nn.Linear(4096, self.hidden_size)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        width = self.embed_dim
        num_layer = 4
        num_head = 8

        self.transformer = Transformer(width, num_layer, num_head)
        self.iter = 1
        self.results = {}
        self.dtype = torch.float16

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def forward(self, results):
        graph_embeddings = []
        self.obj_embed.to(dtype=self.dtype)
        self.rel_embed.to(dtype=self.dtype)
        self.projector.to(dtype=self.dtype)
        self.transformer.to(dtype=self.dtype)

        for result in results:
            triplets = []
            pred_labels = result.get_field('pred_labels')
            rel_pair_idxs = result.get_field('rel_pair_idxs')
            rel_labels = result.get_field('pred_rel_labels').to(self.device)
            pair_pred = torch.stack((pred_labels[rel_pair_idxs[:, 0]], pred_labels[rel_pair_idxs[:, 1]]), dim=1).to(self.device)
            head_emb = self.obj_embed(pair_pred[:, 0])
            rel_emb = self.rel_embed(rel_labels)
            tail_emb = self.obj_embed(pair_pred[:, 1])
            for pair, rel in zip(pair_pred, rel_labels):
                sub_label = self.obj_classes[pair[0].item()]
                obj_label = self.obj_classes[pair[1].item()]
                rel_label = self.rel_classes[rel.item()]
                triplet = sub_label + '_' + rel_label + '_' + obj_label
                triplets.append(triplet)
            # self.results[self.iter] = triplets
            # with open("/home/pcl/sgg_graph_encoder/results.json", "w") as f:
            #     json.dump(self.results, f)
            triple_emb = head_emb + rel_emb + tail_emb
            knowledge_emb = self.transformer(triple_emb.to(dtype=self.dtype))
            knowledge_emb = torch.mean(knowledge_emb, dim=0).unsqueeze(0)
            graph_embeddings.append(knowledge_emb)
            self.iter += 1
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = self.projector(graph_embeddings.unsqueeze(1)).to(torch.bfloat16)
        return graph_embeddings


class SGVisionTower(nn.Module):
    def __init__(self, delay_load=False):
        super().__init__()
        self.sgg_model = None
        self.image_processor = None
        self.is_loaded = False
        self.is_fp16 = False
        self.sg_home = SG_HOME + SG_MODEL
        self.sgg_config_path = SG_HOME + 'pure_sgg_without_csrc/configs/' + SG_YAML
        self.sgg_model_dir = self.sg_home
        self.sgg_model_path = self.sg_home + 'model_final.pth'
        cache_file = torch.load(self.sg_home + 'VG_stanford_filtered_with_attribute_train_statistics.cache')
        self.rel_classes = cache_file['rel_classes']
        self.obj_classes = cache_file['obj_classes']
        self.GLOVE_DIR = SG_HOME + 'glove'

        cfg.merge_from_file(self.sgg_config_path)
        cfg.OUTPUT_DIR = self.sgg_model_dir
        cfg.GLOVE_DIR = self.GLOVE_DIR
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = 'TransformerPredictor'
        cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
        cfg.MODEL.ROI_RELATION_HEAD.RETURN_GRAPH_EMBEDDING = False
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 72 ## 16
        cfg.TEST.CUSTUM_EVAL = True
        cfg.hidden_size = 'sg_size'
        cfg.freeze()
        self.config = cfg
        self.cfg = cfg
        self.sgg_model = build_detection_model(self.cfg)
        self.sgg_model.eval()
        checkpoint = torch.load(self.sgg_model_path, map_location=self.device)
        load_mapping = {}
        load_state_dict(self.sgg_model, checkpoint.pop("model"), load_mapping)
        self.is_loaded = True
        self.dtype = torch.float16

    @torch.no_grad()
    def forward(self, images):
        self.sgg_model.to(dtype=self.dtype)
        self.sgg_model.eval()
        if type(images) is list:
            images = to_image_list(images, 32).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                image_features = self.sgg_model(images)
        else:
            image_forward_outs = self.encode_graphs(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(dtype=images.dtype)

        return image_features

    @property
    def device(self):
        return next(self.sgg_model.parameters()).device

    @property
    def num_patches(self):
        raise ValueError('Scene Graph Encoder has no num_patches')