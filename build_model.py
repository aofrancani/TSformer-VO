import torch
import numpy as np
import os
import torch.nn as nn
from timesformer.models.vit import VisionTransformer
from functools import partial
from einops import rearrange, reduce, repeat
from timesformer.models.helpers import load_pretrained


default_cfgs = {
    "vit_patch16_edim768":
        {
            'url': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth', #'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch16_edim192":
        {
            'url': "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch32_edim1024":
        {
            'url': "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
    "vit_patch16_edim384":
        {
            'url': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            'first_conv': 'patch_embed.proj',
            'classifier': 'head',
        },
}


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args, model_params):
    # build and load model
    model = VisionTransformer(img_size=model_params["image_size"],
                              num_classes=model_params["num_classes"],
                              patch_size=model_params["patch_size"],
                              embed_dim=model_params["dim"],
                              depth=model_params["depth"],
                              num_heads=model_params["heads"],
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_rate=0.,
                              attn_drop_rate=model_params["attn_dropout"],
                              drop_path_rate=model_params["ff_dropout"],
                              num_frames=model_params["num_frames"],
                              attention_type=model_params["attention_type"])

    if model_params["time_only"]:
        # for time former without spatial layers
        for name, module in model.named_modules():
            if hasattr(module, 'attn'):
                # del module.attn
                module.attn = torch.nn.Identity()

    # load checkpoint
    args["epoch_init"] = 1
    args["best_val"] = np.inf
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        args["epoch_init"] = checkpoint["epoch"] + 1
        args["best_val"] = checkpoint["best_val"]
        model.load_state_dict(checkpoint['model_state_dict'])

    elif args["pretrained_ViT"]:  # load ImageNet weights
        img_size = model_params["image_size"]
        num_patches = (img_size[0] // model_params["patch_size"]) * (img_size[1] // model_params["patch_size"])
        model_name = "vit_patch{}_edim{}".format(model_params["patch_size"], model_params["dim"])
        model.default_cfg = default_cfgs[model_name]
    
        print(" --- loading pretrained to start training ---")
        print(model.default_cfg["url"] + "\n")
    
        load_pretrained(model, num_classes=model_params["num_classes"],
                        in_chans=3, filter_fn=_conv_filter, img_size=img_size,
                        num_frames=model_params["num_frames"],
                        num_patches=num_patches,
                        attention_type=model_params["attention_type"],
                        pretrained_model="")


    if torch.cuda.is_available():
        model.cuda()
    
    return model, args


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W
