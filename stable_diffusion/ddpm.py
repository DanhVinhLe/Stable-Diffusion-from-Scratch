import torch
import numpy as np 

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.arange(num_training_steps -1, -1, -1)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
    def set_inference_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def get_previous_timestep(self, timestep: int):
        prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t
    
    def get_variance(self, timestep: int):
        prev_t = self.get_previous_timestep(timestep)
        alpha_cum_t = self.alphas_cumprod[timestep]
        alpha_cum_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=alpha_cum_t.device)
        current_beta_t = 1 - alpha_cum_t / alpha_cum_prev_t
        variance = current_beta_t * (1 - alpha_cum_prev_t) / (1 - alpha_cum_t)
        return variance
    
    def set_strength(self, strength: float = 1.0):
        """
            Set how much noise to add to the input image. 
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self.get_previous_timestep(t)
        alpha_cum_t = self.alphas_cumprod[t]
        alpha_cum_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=alpha_cum_t.device)
        beta_cum_t = 1 - alpha_cum_t
        beta_cum_prev_t = 1 - alpha_cum_prev_t
        current_alpha_t = alpha_cum_t / alpha_cum_prev_t
        current_beta_t = 1 - current_alpha_t
        
        pred_original = (latents - beta_cum_t ** (0.5) * model_output) / alpha_cum_t ** (0.5)
        
        pred_original_coeff = (alpha_cum_prev_t ** (0.5) * current_beta_t) / beta_cum_t
        pred_current_coeff = alpha_cum_t ** (0.5) * beta_cum_prev_t / beta_cum_t
        
        pred_prev_sample = pred_original_coeff * pred_original + pred_current_coeff * latents
        
        variance = 0 
        if t > 0: 
            device = model_output.device
            noise = torch.randn(model_output.shape, device= device, generator= self.generator, dtype = model_output.dtype)
            variance = self.get_variance(t) ** (0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.Tensor):
        alphas_cum = self.alphas_cumprod.to(device = original_samples.device, dtype = original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        noise = torch.randn(original_samples.shape, generator= self.generator, device= original_samples.device, dtype = original_samples.dtype)
        sqrt_alphas_cum = alphas_cum[timesteps] ** (0.5)
        sqrt_one_minus_alphas_cum = (1 - alphas_cum[timesteps]) ** (0.5)
        while(len(sqrt_alphas_cum.shape) < len(noise.shape)):
            sqrt_alphas_cum = sqrt_alphas_cum.unsqueeze(-1)
        while(len(sqrt_one_minus_alphas_cum.shape) < len(noise.shape)):
            sqrt_one_minus_alphas_cum = sqrt_one_minus_alphas_cum.unsqueeze(-1)
        noisy_samples = sqrt_alphas_cum * original_samples + sqrt_one_minus_alphas_cum * noise
        return noisy_samples
        