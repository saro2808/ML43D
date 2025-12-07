import torch
import torch.nn.functional as F


def sample_pdf(bins, weights, num_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / num_samples, 1. - 0.5 / num_samples, steps=num_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, num_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def eikonal_loss(gradients):
    """Eikonal loss
    Args:
        gradients: (num_rays, num_samples_on_ray) gradients of the SDF
    Returns:
        gradient_error: Eikonal loss value (scalar)
    """
    # Implement the eikonal loss
    return torch.mean((torch.norm(gradients, dim=-1) - 1)**2, dim=1).sum()


class NeuSRenderer:
    def __init__(
        self,
        sdf_network,
        color_network,
        variance_network,
        num_samples,
        num_importance,
        up_sample_steps: int = 0,
        perturb: bool = False,
        device='cuda',
    ):
        self.sdf_network = sdf_network
        self.color_network = color_network
        self.variance_network = variance_network
        self.num_samples = num_samples
        self.num_importance = num_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.device = device

    def upsample_along_ray(self, rays_o, rays_d, z_vals, sdf, num_importance, inv_s):
        batch_size, num_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, num_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, num_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=rays_o.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, num_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.get_sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render(self, rays_o, rays_d, near, far):
        z_vals = torch.linspace(0.0, 1.0, self.num_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]

        if self.perturb:
            t_rand = torch.rand(rays_o.shape[0], 1, device=rays_o.device) - 0.5
            z_vals = z_vals + t_rand / self.num_samples

        if self.num_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
                sdf = self.sdf_network.get_sdf(pts.reshape(-1, 3)).reshape(rays_o.shape[0], -1)

                # Use self.up_sample_steps steps to upsample self.num_importance points
                for i in range(self.up_sample_steps):
                    new_z_vals = self.upsample_along_ray(
                        rays_o,
                        rays_d,
                        z_vals,
                        sdf,
                        self.num_importance // self.up_sample_steps,
                        inv_s=64 * (2 ** i),
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o,
                        rays_d,
                        z_vals,
                        new_z_vals,
                        sdf,
                        last=(i == self.up_sample_steps - 1),
                    )

            num_samples_total = self.num_samples + self.num_importance

        render_out = self.render_core(
            rays_o,
            rays_d,
            z_vals,
            num_samples_total,
        )

        return render_out

    def volume_render(self, rgb, alpha):
        """Volume rendering using RGB and alpha values
        Args:
            rgb: (batch_size, num_samples, 3) RGB values
            alpha: (batch_size, num_samples) alpha values
        Returns:
            color: (batch_size, 3) final color
            weights: (batch_size, num_samples) accumulated weights
        """
        # Implement the volume rendering (Eq 11 in the paper)
        T = torch.cat([
            torch.ones((alpha.shape[0], 1), device=self.device),
            torch.cumprod(1 - alpha, dim=1)[:,:-1]
        ], dim=1)
        weights = alpha * T
        color = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        return color, weights

    def render_core(
        self,
        rays_o,
        rays_d,
        z_vals,
        num_samples,
    ):
        """Volume rendering using RGB and SDF values"""
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        sample_dist = 2.0 / num_samples        # Assuming the region of interest is a unit sphere, so the disatance is 2.0 / num_samples
        dists = torch.cat([dists, torch.tensor([sample_dist], device=rays_o.device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        # Query the SDF network
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        # Get the point normal
        gradients = self.sdf_network.gradient(pts).squeeze()        # normal

        # Query the color network
        sampled_color = self.color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = self.variance_network(torch.zeros([1, 3], device=rays_o.device))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        cos_anneal_ratio = 0.0      # For now, we don't use this
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        color, weights = self.volume_render(sampled_color, alpha)
        weight_sum = weights.sum(-1, keepdim=True)

        # Eikonal loss
        gradients = gradients.reshape(batch_size, n_samples, 3)
        gradient_error = eikonal_loss(gradients)

        return {
            'rgb': color,
            'sdf': sdf,
            'gradient_error': gradient_error,
            'weight_sum': weight_sum,
        }

    @torch.no_grad()
    def extract_fields(self, bound_min, bound_max, resolution, batch_size):
        """Extracts the SDF field from the network (from implicit to explicit)
        Args:
            bound_min: minimum bound of the field
            bound_max: maximum bound of the field
            resolution: resolution of the field
            batch_size: number of points that are input to the network at once
        Return:
            sdf_grid: SDF field in a grid format with shape (resolution, resolution, resolution)
        """
        x = torch.linspace(bound_min[0], bound_max[0], resolution)
        y = torch.linspace(bound_min[1], bound_max[1], resolution)
        z = torch.linspace(bound_min[2], bound_max[2], resolution)

        xx, yy, zz = torch.meshgrid(x, y, z)
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        sdf_grid = torch.zeros(resolution * resolution * resolution)
        # Query the SDF values in batch (cannot fit all points at once in GPU memory)
        for i in range(0, len(xyz), batch_size):
            pts = xyz[i:i + batch_size].to(self.device)
            sdf = self.sdf_network.get_sdf(pts)
            sdf_grid[i:i + batch_size] = sdf[:, 0].cpu()

        sdf_grid = sdf_grid.reshape(resolution, resolution, resolution)
        return sdf_grid
