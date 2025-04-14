import torch
import numpy as np 
import torch.nn as nn 
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def generate(prompt: str, uncond_prompt: str, input_images = None,
            strength = 0.8, do_cfg = True, cfg_scale = 7.5, 
            sampler_name = "ddpm", n_inference_step = 50, models = {}, seed = None,
            device = None, idle_device = "cpu", tokenizer = None):
    with torch.no_grad():
        if not (0 < strength and strength < 1):
            raise ValueError("Strength must be between 0 and 1")
        # idle_device to avoid OOM error of GPU
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert prompt to tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            # (B, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device = device)
            # (B, seq_len) -> (B, seq_len, d_embed)
            cond_tokens = clip(cond_tokens)
            
            uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids, dtype = torch.long, device = device)
            uncond_tokens = clip(uncond_tokens)
            
            context = torch.cat([cond_tokens, uncond_tokens]) # (2* B, seq_len, d_embed)
        
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device = device)
            context = clip(tokens)
            
        clip.to(idle_device)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_step)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        if input_images:
            encoder = models["encoder"]
            encoder.to(device)
            input_images_tensor = input_images.resize((WIDTH, HEIGHT))
            input_images_tensor = np.array(input_images_tensor)
            input_images_tensor = torch.tensor(input_images_tensor, dtype = torch.float32, device = device) # (H, W, C)
            input_images_tensor = rescale(input_images_tensor, (0, 255), (-1, 1))
            input_images_tensor = input_images_tensor.unsqueeze(0).permute(0, 3, 1, 2) # (C, H, W) -> (B, C, H, W)
            encoder_noise = torch.randn(latent_shape, generator= generator, device= device)
            latents = encoder(input_images_tensor, encoder_noise) # (B, 4, H /8, W/8)
            sampler.set_strength(strength = strength)
            
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            # to_idle(encoder)
            encoder.to(idle_device)
        else:
            latents = torch.randn(latent_shape, generator= generator, device= device)
            
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device) # (B, 320)
            model_input = latents 
            if do_cfg: 
                model_input = model_input.repeat(2, 1, 1, 1) # (2*B, 4, H /8, W/8)
            model_output = diffusion(model_input, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim = 0)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            latents = sampler.step(timestep, latents, model_output)
        
        diffusion.to(idle_device)
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        diffusion.to(idle_device)
        images = rescale(images, (-1, 1), (0, 255), clamp= True)
        # (B, C, H, W) -> (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
            