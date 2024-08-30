import torch
import torch.nn as nn
import torch.nn.functional as F

class IdempotentLoss(nn.Module):

    def __init__(self, model, rec_weight, idem_weight, tight_weight, loss_tight_clamp_ratio, batch_size):
        super().__init__()
        self.training_model_copy = model
        self.rec_weight = rec_weight
        self.idem_weight = idem_weight / rec_weight
        self.tight_weight = tight_weight / rec_weight
        self.loss_tight_clamp_ratio = loss_tight_clamp_ratio
        self.batch_size = batch_size


    def forward(self, input_data, output_data, training_model):
        cur_batch_size = input_data.shape[0]
        # Reconstruction loss
        loss_rec = F.mse_loss(input_data, output_data, reduction='none').reshape(cur_batch_size, -1).mean(dim=-1)

        idem_data = input_data.permute(0, 2, 1)  # batch_size * time_dimension * length
        self.training_model_copy.load_state_dict(training_model.state_dict())
        freq_means_and_stds = torch.stack(self.get_freq_means_and_stds(idem_data)).unsqueeze(0)
        num_dims = len(freq_means_and_stds.shape) - 1
        freq_means_and_stds = freq_means_and_stds.repeat(idem_data.shape[0], *(1,) * num_dims).unbind(dim=1)
        z = self.get_noise(*freq_means_and_stds)
        z = z.permute(0, 2, 1)  # batch_size * length * dimension
        fz, _, _, _ = training_model(z)
        f_z = fz.detach()
        ff_z, _, _, _ = training_model(f_z)
        f_fz, _, _, _ = self.training_model_copy(fz)

        # Idempotent loss
        loss_idem = F.l1_loss(f_fz, fz, reduction='mean')

        # Tightness loss
        loss_tight = -F.l1_loss(ff_z, f_z, reduction='none').reshape(cur_batch_size, -1).mean(dim=-1)
        loss_tight_clamp = self.loss_tight_clamp_ratio * loss_rec
        loss_tight = F.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp
        loss_rec = loss_rec.mean()
        loss_tight = loss_tight.mean()

        # Get total loss 'without' reconstruction loss
        loss = self.idem_weight * loss_idem + self.tight_weight * loss_tight
        return loss

    def get_freq_means_and_stds(self, x):
        freq = torch.fft.fft(x, dim=-1)
        real_mean = freq.real.mean(dim=0)
        real_std = freq.real.std(dim=0)
        imag_mean = freq.imag.mean(dim=0)
        imag_std = freq.imag.std(dim=0)
        return real_mean, real_std, imag_mean, imag_std

    def get_noise(self, real_mean, real_std, imag_mean, imag_std):
        freq_real = torch.normal(real_mean, real_std)
        freq_imag = torch.normal(imag_mean, imag_std)
        freq = freq_real + 1j * freq_imag
        noise = torch.fft.ifft(freq, dim=-1)
        return noise.real
