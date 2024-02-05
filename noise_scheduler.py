
import torch

class LinearNoiseScheduler:
    """
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

        # p2_k = 1
        # p2_gamma = 0.5 #1.0 
        # self.lambda_t = ((1 - self.betas) * (1 - self.alpha_cum_prod))/self.betas
        # self.snr =  (1.0/(1.0 - self.alpha_cum_prod)) - 1  #alpha_cum_prod/(1.0 - alpha_cum_prod) 
        # self.weights = (self.lambda_t/(p2_k + self.snr)**p2_gamma)
        # self.norm_weights = self.weights/torch.max(self.weights)


    # def get_pp_weights(self, t, normalize=False):
    #     """
    #     Return perception prioritized weights for the given time steps.
    #     https://arxiv.org/pdf/2204.00227.pdf

    #     :param t: timestep of the forward process of shape -> (B,)
    #     :return:  weights with shape (B, 1, 1, 1)
    #     """
        
    #     if normalize:
    #         weights = self.norm_weights[t]
    #     else:
    #         weights = self.weights[t]

    #     # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
    #     # Assume unsqueeze 3 times to turn (B) -> (B, 1, 1, 1)
    #     for _ in range(3):
    #         weights = weights.unsqueeze(-1)

    #     return weights


    def add_noise(self, original, noise, t):
        """
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape)-1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
        

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / torch.sqrt(self.alpha_cum_prod[t])
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t])*noise_pred)/(self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, mean
        else:
            variance = (1-self.alpha_cum_prod[t-1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0