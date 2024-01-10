# From https://github.com/carolineec/informative-drawings
# MIT License

import os
import cv2
from PIL import Image
import torch
import numpy as np
import argparse
from torchvision.transforms import ToPILImage
from torchvision import transforms
import gradio as gr
from enum import Enum

import torch.nn as nn
from einops import rearrange

MARKDOWN_INFO = "<font size='0.5'> 风格模式设置了5种档位, 风格强度从弱到强分别是：<br> \
                '1.Clasic Elegance': ip scale 建议选取0.8~2.0 <br> \
                '2.Artistic Fusion': ip scale 建议选取0.6~1.4 <br> \
                '3.Expressive Vibe': ip scale 建议选取0.4~1.2 <br> \
                '4.Bold Transformation': ip scale 建议选取0.2~0.5 <br> \
                '5.Fantasy World': ip scale 建议选取0.2~0.5 <br> \
                <font size='0.2'>ps:上面推荐参数是针对 plus 版本单图推理的设置</font><br> \
                上述5种风格(0-5)  数字越大风格越强, 对ip scale越敏感, 对结构的改变越夸张 <br> \
                多图模式对ipscale不敏感 ——> 可以适当加强ipscale </font>\ "

def set_parser():
    source_dir = r"/mnt/nfs/file_server/public/mingjiahui/models"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model_path', type=str,
        default='/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
    )
    parser.add_argument(
        '--vit_h', type=str, help=r'default is vit_h',
        default=f'{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder/',
    )
    parser.add_argument(
        '--vit_g', type=str, help=r'default is vit_g',
        default=f'{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/image_encoder/',
    )
    parser.add_argument(
        '--ip_ckpt', type=str,
        default=f'{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin'
    )
    parser.add_argument(
        '--controlnet_model_path', type=str,
        default=f'{source_dir}/diffusers--controlnet-depth-sdxl-1.0/'
    )
    parser.add_argument(
        '--port', type=int, default=10050
    )
    parser.add_argument(
        '--token_num', type=int, default=16
    )

    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self):
        self.model = self.load_model('sk_model.pth')
        self.model_coarse = self.load_model('sk_model2.pth')
        self.to_pil = ToPILImage()

    def load_model(self, name):
        modelpath = os.path.join("/mnt/nfs/file_server/public/lipengxiang/ALL_CODE/diffusers/examples/controlnet", name)
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        model.eval()
        model = model.cuda()
        return model

    def __call__(self, input_image: Image.Image, coarse=False):
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        assert input_image.ndim == 3

        model = self.model_coarse if coarse else self.model
        
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().cuda()
            res = model(image.permute(2, 0, 1))

        return self.to_pil(res)


from transformers import DPTFeatureExtractor, DPTForDepthEstimation
depth_model_path = "/mnt/nfs/file_server/public/mingjiahui/models/Intel--dpt-hybrid-midas"
depths_processor = DPTForDepthEstimation.from_pretrained(depth_model_path).to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained(depth_model_path)
lineart_processor = LineartDetector()  


def get_depths(image: Image.Image):
    image = feature_extractor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depths_processor(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def get_lineart(image: Image.Image):
    image = image.convert("RGB")
    return lineart_processor(image)


def get_canny(image: Image.Image, canny_low=100, canny_high=200):
    image = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
    image = cv2.Canny(image, canny_low, canny_high)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


class IPAdapterUi:
    def __init__(self, model_dir, tabname):
        self.model_dir = model_dir
        self.unit_group = self.ui_group(tabname)
        self.unit_id = None
    
    # def ip_model_update(self, model_id):
    #     if model_id == 'None':
    #         return model_id
        
    #     from ui_v2 import model_dir, ip_model, args
    #     ip_ckpt = os.path.join(model_dir, model_id)
    #     image_encoder_path = args.vit_h if 'vit-h' in model_id else args.vit_g
    #     token_num = 16 if 'plus' in ip_ckpt else 4

    #     ip_model.update_unit(
    #         self.unit_id, 
    #         param_dict={
    #             'image_encoder_path': image_encoder_path,
    #             'ip_ckpt': ip_ckpt,
    #             'token_num': token_num,
    #         })

    #     return model_id

    def ui_group(self, tn):
        self.unit_id = int(tn.split('Unit')[1])
        group_set = set()
        with gr.Group() as group:
            # # debug
            # enable = gr.Checkbox(elem_id=tn)
            # debug_textbox = gr.Textbox(elem_id=tn)
            # group_set = {enable, debug_textbox}
            
            with gr.Column(variant='compact'):
                model_ids = [name for name in os.listdir(self.model_dir) if name.endswith('.bin')]+['None']
                model_id = gr.Dropdown(
                    choices=model_ids, 
                    label='ip model', 
                    elem_id=f'{tn}-model_id',
                    value='ip-adapter-plus_sdxl_vit-h.bin' if self.unit_id==0 else 'None',
                    )
                # model_id.change(fn=self.ip_model_update, inputs=model_id, outputs=model_id)
            group_set = group_set.union({model_id})

            with gr.Tab("Single Reference"):
                single_enable = gr.Checkbox(elem_id=f'{tn}-single_enable', label='enable')
                with gr.Column(variant='compact'):
                    style_image = gr.Image(label='input', type='pil', elem_id=f'{tn}-style_image', )
                    ip_scale = gr.Slider(minimum=0.0, maximum=4.0, step=0.1, label="ip scale", value=0.3, elem_id=f"{tn}-ip_scale")
                    add_mode = gr.Dropdown(
                        [
                            '1.Clasic Elegance    (last5layers)',
                            '2.Artistic Fusion    (front5layers)', 
                            '3.Expressive Vibe    (10layers)', 
                            '4.Bold Transformation    (upblocks-only)', 
                            '5.Fantasy World  (full)', ], 
                        label='style strength mode', 
                        value='5.Fantasy World  (full)',
                        elem_id=f'{tn}-add_mode',
                        )
                    # single_output = gr.Image(label='output', type='pil')
                # single_button = gr.Button("Submit")
            group_set = group_set.union({single_enable, style_image, ip_scale, add_mode})

            with gr.Tab("Multi Reference"):
                multi_enable = gr.Checkbox(elem_id=f'{tn}-multi_enable', label='enable')
                with gr.Row(variant='compact'):
                    with gr.Column(scale=1):
                        gr.Markdown("Different style")
                        structure_image = gr.Image(label='structure', type='pil', elem_id=f'{tn}-structure_image')
                        color_image = gr.Image(label='color', type='pil', elem_id=f'{tn}-color_image')
                        is_pro =  gr.Checkbox(label="pro", info="if checked, it will enhance the style, but there is a risk of structural collapse", elem_id=f'{tn}-is_pro')
                        with gr.Row(variant='compact'):
                            structure_scale = gr.Slider(minimum=0.0, maximum=1.2, step=0.1, label="structure ip scale", value=0.4, elem_id=f'{tn}-structure_scale')
                            color_scale = gr.Slider(minimum=0.0, maximum=1.2, step=0.1, label="color ip scale", value=1.0, elem_id=f'{tn}-color_scale')
                        gr.Markdown("same style")
                        image_paths = gr.File(file_count='multiple', elem_id=f'{tn}-image_paths')
                        with gr.Row(variant='compact'):
                            cache_path = gr.Textbox(label='Cache File Path', scale=10, elem_id=f'{tn}-cache_path')
                            cache_refresh = gr.Button(value=UiSymbol.refresh_symbol, size='sm', min_width=1, variant='tool', scale=1)
                            cache_state = gr.Textbox(label='check exist', scale=2)
                            def check_exist(cache_path):
                                return os.path.exists(cache_path)
                            cache_refresh.click(fn=check_exist, inputs=[cache_path], outputs=[cache_state])
                            
                        with gr.Row(variant='compact'):
                            multi_ip_scale = gr.Slider(minimum=0.0, maximum=4.0, step=0.1, label="ip scale", value=0.3, elem_id=f'{tn}-multi_ip_scale', scale=1)
                            multi_add_mode = gr.Dropdown(
                                [
                                    '1.Clasic Elegance    (last5layers)',
                                    '2.Artistic Fusion    (front5layers)', 
                                    '3.Expressive Vibe    (10layers)', 
                                    '4.Bold Transformation    (upblocks-only)', 
                                    '5.Fantasy World  (full)', 
                                    ], 
                                label='style strength mode', 
                                value='5.Fantasy World  (full)', 
                                elem_id=f'{tn}-multi_add_mode',
                                scale=2
                                )
                    # with gr.Column(scale=1):
                    #     multi_output = gr.Image()
                # multi_button = gr.Button("Submit")
            group_set = group_set.union({multi_enable, structure_image, color_image, is_pro, structure_scale, color_scale, image_paths, multi_ip_scale, multi_add_mode, cache_path})

            with gr.Accordion("Ip Scale Info", open=False):
                gr.Markdown(MARKDOWN_INFO)

            with gr.Accordion("Other Ip Control Param", open=False):
                with gr.Accordion("Resize Mode", open=False):
                    resize_mode = gr.Radio(
                        choices=[ImageOperation.JUST_RESIZE, ImageOperation.CROP_RESIZE],
                        value=ImageOperation.CROP_RESIZE,
                        # label="Resize Mode", 
                        show_label=False,
                        elem_id=f'{tn}-resize_mode',
                        )
                with gr.Accordion("Denoising Control", open=False):
                    start_control_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Starting Control Step", value=0.0, elem_id=f"{tn}-start_control_step")
                    ending_control_step = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Ending Control Step", value=1.0, elem_id=f"{tn}-end_control_step")
                with gr.Accordion("IP-KV Norm", open=False):
                    cn_weights = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, show_label=False, elem_id=f"{tn}-cn_weights")
            group_set = group_set.union({resize_mode, start_control_step, ending_control_step, cn_weights})
                    
        return model_id, group_set


class ControlMode(Enum):
    """
    The improved guess mode.
    """
    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"


class ImageOperation(Enum):
    JUST_RESIZE = "just_resize"
    CROP_RESIZE = "crop_and_resize"
    OUTPUT_RESIZE = "output_resize"
    @staticmethod
    def just_resize(img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.resize((512, 512))
    @staticmethod
    def crop_and_resize(img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        return transform(img)
    @staticmethod
    def output_reszie(img, w, h):
        return img.resize((w, h))
    

class OtherTrick(Enum):
    IP_KV_NORM = 'Ip-KV Norm'
    UNCOND_IMG_EMBEDS_CACHE = 'Fooocus-UncondImagEmbeds Cache'
    UNCOND_IMG_EMBEDS_CACHE_PATH = r'./data/fooocus_ip_negative.safetensors'
    GRAY_UNCOND_IMG_EMBEDS = 'Gray-UncondImageEmbeds'


class LoRA(Enum):
    # # ti_lora
    # TILoRA_CuteFelt = r'/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/XL版本上线lora/CuteFelt_SDXL_v0.safetensors'
    # TILoRA_CuteFelt_TI = r'/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/embeddings/Cute_Felt_xl_TI.safetensors'

    # lora_dir1 = r'/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/iplora/21Lora/'
    # lora_dir2 = r'/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/iplora/20Lora/'
    # iplora_face_plus = r'/mnt/nfs/file_server/public/mingjiahui/models/lora/iplora-face_plus/'

    lora_dir1 = r"/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/iplora/lora_official/"


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""
    def __init__(self, **kwargs):
        super().__init__(elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)
    def get_block_name(self):
        return "button"


class UiSymbol(object):
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝

    tooltips = {
        '🔄': 'Refresh',
        '\u2934': 'Send dimensions to stable diffusion',
        '💥': 'Run preprocessor',
        '📝': 'Open new canvas',
        '📷': 'Enable webcam',
        '⇄': 'Mirror webcam',
    }