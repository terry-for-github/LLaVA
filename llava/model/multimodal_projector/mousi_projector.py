from typing import List
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..multimodal_encoder.graph_encoder import SGAdapter


class MousiProjector(nn.Module):
    def __init__(self, image_hidden_size_list: List[int], m_token_one_patch: List[int], llm_hidden_size: int):
        super().__init__()

        self.mlp1_list = nn.ModuleList()
        self.m_list = m_token_one_patch
        # print('image_hidden_size_list, llm_hidden_size: ', image_hidden_size_list, llm_hidden_size)
        for i, image_hidden_size in enumerate(image_hidden_size_list):
            # special judge for sg encoder
            if image_hidden_size == 'sg_size':
                self.mlp1_list.append(SGAdapter())
            else:
                self.mlp1_list.append(nn.Linear(image_hidden_size * self.m_list[i], llm_hidden_size))
        self.mlp2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, image_features_list: List[torch.Tensor]):
        hidden_features_list = []
        # print('projector before:', list(map(lambda x: x.shape, image_features_list)))
        for i, image_features in enumerate(image_features_list):
            # m-patches-one-token
            # image_features: bs, num_patches, hidden_size
            # graph_encoder'm must be 1
            if type(image_features) == torch.Tensor and self.m_list[i] > 1:
                bs, num_patches, _ = image_features.shape
                image_features = image_features.view(bs, num_patches // self.m_list[i], -1)
            hidden_features_list.append(self.mlp1_list[i](image_features))
        # print('projector after:', list(map(lambda x: x.shape, hidden_features_list)))
        hidden_feature = torch.cat(hidden_features_list, dim=1)
        # print('projector final:', hidden_feature.shape)
        return self.mlp2(F.gelu(hidden_feature))


