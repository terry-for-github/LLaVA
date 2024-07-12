import copy
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers import CLIPVisionModel, LayoutLMv3Model, Dinov2Model
from transformers import BitImageProcessor, LayoutLMv3ImageProcessor, CLIPImageProcessor
from .graph_encoder import SGVisionTower, GraphProcessor


class MoEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.experts_list = args.vision_experts_list
        # print(self.experts_list)
        self.encoders_list = nn.ModuleList()

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        # else:
        #     self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = MoEImageProcessor(self.experts_list)

        """
            ValueError: ***Model does not support `device_map='auto'`.
            To implement support, the model class needs to implement the `_no_split_modules` attribute.
            So I set _no_split_modules attribute.
        """
        for expert in self.experts_list:
            if expert == 'openai/clip-vit-large-patch14-336':
                # Cant use AutoModel because of CLIPModel
                # defaults setting
                # CLIPVisionModel._no_split_modules = ['CLIPEncoderLayer']
                self.encoders_list.append(CLIPVisionModel.from_pretrained(expert, device_map=device_map))
            elif expert == 'facebook/dinov2-giant':
                Dinov2Model._no_split_modules = ["Dinov2Layer"]
                self.encoders_list.append(AutoModel.from_pretrained(expert, device_map=device_map))
            elif expert == 'facebook/dinov2-large':
                Dinov2Model._no_split_modules = ["Dinov2Layer"]
                self.encoders_list.append(AutoModel.from_pretrained(expert, device_map=device_map))
            elif expert == 'microsoft/layoutlmv3-large':
                LayoutLMv3Model._no_split_modules = ["LayoutLMv3Layer"]
                self.encoders_list.append(AutoModel.from_pretrained(expert, device_map=device_map))
            elif expert == 'graph_encoder':
                graph_encoder = SGVisionTower()
                graph_encoder.eval()
                self.encoders_list.append(graph_encoder)
            else:
                raise NotImplementedError(expert + ' encoder not implemented.')
            # print(self.encoders_list[-1], self.encoders_list[-1]._no_split_modules)
        self.encoders_list.requires_grad_(False)
        # for encoder in self.encoders_list:
        #     encoder = encoder.eval()
        # print(list(map(lambda encoder: type(encoder), self.encoders_list)))
        # print(self.encoders_list)

        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        '''
        images: [[image1_clip, image1_dino, image1_ocr], [image2_clip, image2_dino, image2_ocr], ...] all Tensor
        '''
        # encoder_image_list: [[image1_clip, image2_clip, ...], [image1_dino, image2_dino, ...], [image1_ocr, image2_ocr, ...]]
        encoder_image_list = [[image[i] for image in images] for i in range(len(images[0]))]
        image_features_list = []
        devices = self.device
        dtypes = self.dtype
        assert len(encoder_image_list) == len(self.experts_list), (len(encoder_image_list), len(self.experts_list))
        for i, expert in enumerate(self.experts_list):
            if expert == 'openai/clip-vit-large-patch14-336':
                image = torch.stack(encoder_image_list[i]).to(device=devices[i], dtype=dtypes[i])
                image_forward_outs = self.encoders_list[i](image, output_hidden_states=True)
            elif expert == 'facebook/dinov2-giant':
                image = torch.stack(encoder_image_list[i]).to(device=devices[i], dtype=dtypes[i])
                image_forward_outs = self.encoders_list[i](pixel_values=image, output_hidden_states=True)
            elif expert == 'facebook/dinov2-large':
                image = torch.stack(encoder_image_list[i]).to(device=devices[i], dtype=dtypes[i])
                image_forward_outs = self.encoders_list[i](pixel_values=image, output_hidden_states=True)
            elif expert == 'microsoft/layoutlmv3-large':
                image = torch.stack(encoder_image_list[i]).to(device=devices[i], dtype=dtypes[i])
                image_forward_outs = self.encoders_list[i](pixel_values=image, output_hidden_states=True)
            elif expert == 'graph_encoder':
                image_features_graph = self.encoders_list[i](encoder_image_list[i])
                image_features_list.append(image_features_graph)
                continue
            else:
                raise NotImplementedError(expert + ' encoder not implemented.')
            # image_forward_outs: bs * num_tokens * hidden_size
            # print('expert:', expert, len(image_forward_outs.hidden_states), image_forward_outs.hidden_states[-2].shape, self.select_layer)
            image_features = self.feature_select(image_forward_outs).to(dtypes[i])
            image_features_list.append(image_features)

        return image_features_list
    
    @property
    def dtype(self):
        return [encoder.dtype for encoder in self.encoders_list]

    @property
    def device(self):
        return [encoder.device for encoder in self.encoders_list]

    @property
    def hidden_size(self):
        return [encoder.config.hidden_size for encoder in self.encoders_list]


class MoEImageProcessor:
    def __init__(self, experts_list):
        self.processors_list = []
        self.experts_list = experts_list
        for expert in self.experts_list:
            if expert == 'openai/clip-vit-large-patch14-336':
                self.processors_list.append(AutoImageProcessor.from_pretrained(expert))
                self.processors_list[-1].do_convert_rgb = False
            elif expert == 'facebook/dinov2-giant':
                self.processors_list.append(AutoImageProcessor.from_pretrained(expert))
                self.processors_list[-1].do_convert_rgb = False
            elif expert == 'facebook/dinov2-large':
                self.processors_list.append(AutoImageProcessor.from_pretrained(expert))
                self.processors_list[-1].do_convert_rgb = False
            elif expert == 'microsoft/layoutlmv3-large':
                self.processors_list.append(AutoImageProcessor.from_pretrained(expert))
                self.processors_list[-1].apply_ocr = False
            elif expert == 'graph_encoder':
                self.processors_list.append(GraphProcessor())
            else:
                raise NotImplementedError(expert + ' encoder not implemented.')
            # print(expert, self.processors_list[-1])
        # print(list(map(lambda processor: type(processor), self.processors_list)))
    
    def help_expand2square(self, image, expand2square_func):
        image_list = []
        for processor in self.processors_list:
            image_list.append(expand2square_func(image, tuple(int(x*255) for x in processor.image_mean)))
        # print(list(map(lambda image: id(image), image_list)))
        return image_list

    def help_copy_image(self, image):
        return [copy.deepcopy(image) for _ in self.processors_list]

    def help_dummy_image(self):
        dummy_image_list = []
        for i, expert in enumerate(self.experts_list):
            if expert == 'openai/clip-vit-large-patch14-336':
                crop_size = self.processors_list[i].crop_size
            elif expert == 'facebook/dinov2-giant':
                crop_size = self.processors_list[i].crop_size
            elif expert == 'facebook/dinov2-large':
                crop_size = self.processors_list[i].crop_size
            elif expert == 'microsoft/layoutlmv3-large':
                crop_size = self.processors_list[i].size
            elif expert == 'graph_encoder':
                crop_size = self.processors_list[i].crop_size
            else:
                raise NotImplementedError(expert + ' encoder not implemented.')
            dummy_image_list.append(torch.zeros(3, crop_size['height'], crop_size['width']))
        return dummy_image_list

    
    def preprocess(self, image_list):
        processed_image_list = []
        assert len(image_list) == len(self.processors_list), (len(image_list), len(self.processors_list))
        for i, processor in enumerate(self.processors_list):
            out = processor.preprocess(image_list[i], return_tensors='pt')
            if not isinstance(out, torch.Tensor):
                processed_image_list.append(out['pixel_values'][0])
            else:
                processed_image_list.append(out)
        return processed_image_list