import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map, target_size):
    """
    attn_map: {head_num, 4096, dim}
    return: {head, 512, 512}
    """
    # jiahui's modify
    if len(attn_map.shape) == 3:
        head, token_length, head_dim = attn_map.shape
        attn_map = attn_map.permute(2, 0, 1)  # {head_dim, head, 4096}
    elif len(attn_map.shape) == 2:
        attn_map = attn_map
    else:
        ValueError("some error happened")
    
    attn_map = torch.mean(attn_map, dim=0)  # {head, 4096}
    temp_size = None

    for i in range(3,7):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]:
            temp_size = (target_size[0]//scale, target_size[1]//scale)
            break
    assert temp_size is not None, ValueError("temp_size cannot is None")

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False,
    )[0]

    # head, h, w = attn_map.shape
    # attn_map = torch.softmax(attn_map.view(head, -1), dim=1).view((head, h, w))
    # assert int(attn_map[0, :, :].sum().item())==1, ValueError("softmax process is error, please check it")
    return attn_map


# def upscale(attn_map, target_size):
#     """
#     attn_map: {head_num, 4096, dim}
#     """
#     attn_map = torch.mean(attn_map, dim=0)      # attn_map: {head_num, 4096, dim}
#     attn_map = attn_map.permute(1,0)        # attn_map: {4096, head_num, dim}
#     temp_size = None

#     for i in range(0,5):
#         scale = 2 ** i
#         if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
#             temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
#             break

#     assert temp_size is not None, "temp_size cannot is None"

#     attn_map = attn_map.view(attn_map.shape[0], *temp_size)

#     attn_map = F.interpolate(
#         attn_map.unsqueeze(0).to(dtype=torch.float32),
#         size=target_size,
#         mode='bilinear',
#         align_corners=False
#     )[0]

#     attn_map = torch.softmax(attn_map, dim=0)
#     return attn_map

def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.stack(net_attn_maps,dim=0)    # {16,8,512,512} 
    net_attn_maps_mean = torch.mean(net_attn_maps,dim=1)    # {16,512,512}
    return net_attn_maps, net_attn_maps_mean

def attnmaps2images(net_attn_maps):
    """
    jiahui's remark
    net_attn_maps: {8,512,512}
    return : atten map group list-> element:{512,512}
    """
    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")
