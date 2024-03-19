from typing import Union
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import cv2

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torchvision.models import resnet50, ResNet50_Weights


class IdentityDiscriminator(nn.Module):
    def __init__(self, 
            disc_start, 
            logvar_init=0.0, 
            # kl_weight=1.0,
            disc_num_layers=3, 
            disc_in_channels=3, 
            disc_factor=1.0, 
            disc_weight=1.0,
            # perceptual_weight=1.0, 
            disc_loss="hinge",
            disc_type="PatchGAN",       # Union["PatchGAN", "ResNet50"]
            ):
        """
        disc_start:     start iterations of discriminator loss is applied to affect the weight of GAN loss.
        logvar_init:        The initial value of the logarithmic variance;
                                which is used to measure reconstruction losses and canonical losses
        kl_weight:      kl loss weight
        disc_num_layers:        discriminator layers num
        disc_in_channels:       discriminator input channels
        disc_factor:            
        disc_weight:
        perceptual_weight:
        """

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        # self.kl_weight = kl_weight
        # self.perceptual_loss = LPIPS().eval()
        # self.perceptual_weight = perceptual_weight
        self.disc_in_channels = disc_in_channels

        # output log variance
        # self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        if disc_type == 'PatchGAN':
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                    n_layers=disc_num_layers,
                                                    ).apply(weights_init)
        elif disc_type == 'ResNet50':
            self.discriminator = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.discriminator.fc = nn.Linear(2048, 2)

        self.discriminator.train()
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def latents_to_rgb(self, latents):
        weights = (
            (60, -60, 25, -70),
            (60,  -5, 15, -50),
            (60,  10, -5, -35)
        )

        weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
        biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy()
        image_array = image_array.transpose(1, 2, 0)  # Change the order of dimensions
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image_array)
    

    def forward(self, 
            target_noise, # inputs, 
            noise_pred, # reconstructions, 
            noisy_latents,
            t,
            noise_scheduler,
            # posteriors, 
            optimizer_idx,
            global_step, 
            last_layer=None, 
            split="train",
            lantent_type="prev",     # Union["origin", ""prev"]
            ):
        assert target_noise.shape[1] == self.disc_in_channels, \
            ValueError("The input channels of Discriminator and the output channels of diffusion predicted are different")

        # reconstruction loss
        ## ori method
        rec_loss = nn.functional.mse_loss(noise_pred.float(), target_noise.float(), reduction="mean")

        ## other method
        # rec_loss = torch.abs(target_noise.contiguous() - noise_pred.contiguous())
        # # for LPIPS loss 
        # # if self.perceptual_weight > 0:
        # #     p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        # #     rec_loss = rec_loss + self.perceptual_weight * p_loss

        # # Let the model learn which regions are more difficult to reconstruct, 
        # # and reduce the effect of reconstruction errors by increasing the logvar of that region
        # nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # weighted_nll_loss = nll_loss
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # rec_loss = nll_loss

        latents_gt, latents_pred = [], []
        for target_noise_, noise_pred_, t_, noisy_latents_ in zip(target_noise, noise_pred, t, noisy_latents):
            if lantent_type == 'prev':
                prev_latents_gt_ = noise_scheduler.step(target_noise_, t_, noisy_latents_).prev_sample
                prev_latents_pred_ = noise_scheduler.step(noise_pred_, t_, noisy_latents_).prev_sample
            elif lantent_type == 'origin':
                prev_latents_gt_ = noise_scheduler.step(target_noise_, t_, noisy_latents_).pred_original_sample
                prev_latents_pred_ = noise_scheduler.step(noise_pred_, t_, noisy_latents_).pred_original_sample
                
            latents_gt.append(prev_latents_gt_)
            latents_pred.append(prev_latents_pred_)
            #         x = noise_scheduler.step(noise_pred, t, x).prev_sample

        latents_gt = torch.stack(latents_gt)
        latents_pred = torch.stack(latents_pred)
        
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(latents_pred.contiguous())
            g_loss = -torch.mean(logits_fake)

            # method 0
            # c = 100 / (t + 1)
            c = 1
            loss = rec_loss + 0.1 * g_loss

            # log = f"total_loss:{loss.clone().detach().mean()} rec_loss:{rec_loss.detach().mean()} g_loss:{g_loss.detach().mean()}"
            log = {
                "total_loss": loss.clone().detach().mean(),
                "rec_loss": rec_loss.detach().mean(),
                "g_loss": g_loss.detach().mean(),
            }

            # debug
            debug_image = {
                # 'prev_latents_gt': to_pil_image(latents_gt[-1]*0.5+0.5),
                'prev_rgb_gt': self.latents_to_rgb(latents_gt[-1]),
                # 'prev_latents_pred': to_pil_image(latents_pred[-1]*0.5+0.5),
                'prev_rgb_pred': self.latents_to_rgb(latents_pred[-1]),
                'logits_fake': to_pil_image(logits_fake[-1] * 255),
                't': t[-1].item()
            }

            return loss.mean(), log, debug_image

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(latents_gt.contiguous().detach())
            logits_fake = self.discriminator(latents_pred.contiguous().detach())


            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            # log = f"disc_loss:{d_loss.clone().detach().mean()} logits_real:{logits_real.detach().mean()} logits_fake:{logits_fake.detach().mean()}"
            log = {
                "disc_loss": d_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
            }

            return d_loss, log, None

