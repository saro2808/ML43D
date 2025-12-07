from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh
from PIL import Image
import numpy as np

from exercise_3.model.renderer import NeuSRenderer
from exercise_3.model.neus import SDFField, ColorField, SingleVarianceNetwork, compute_psnr
from exercise_3.data.dtu import DTUDataset, DTUTrainDataset, DTUValDataset


def train(
    sdf_field: torch.nn.Module,
    color_field: torch.nn.Module,
    variance_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader:  torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    device,
    save_path: Path,
    config: Dict[str, Any],
    start_epoch: int = 0,
):
    # Declare the renderer
    renderer = NeuSRenderer(
        sdf_field,
        color_field,
        variance_network,
        num_samples=config["num_samples_init"],
        num_importance=config["num_samples_importance"],
        up_sample_steps=4,
        perturb=False,
        device=device,
    )

    # Keep track of the loss
    train_loss_running = 0.0
    train_psnr_running = 0.0
    best_loss = float('inf')

    for epoch in range(start_epoch, config['max_epochs']):

        for batch_idx, batch in enumerate(train_dataloader):
            # Zero out previously accumulated gradients
            optimizer.zero_grad()

            # Move batch to device
            batch = DTUDataset.move_to_device(batch, device)
            rays_o, rays_d, rgb_gt, mask = batch

            # The first two dimensions should be (num_image_per_batch, num_rays_per_image)
            # Therefore, we need to flatten the first two dimensions
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            rgb_gt = rgb_gt.view(-1, 3)
            mask = mask.view(-1, 1)

            # Forward pass
            near, far = DTUDataset.near_far_from_sphere(rays_o, rays_d)
            near, far = near.to(device), far.to(device)

            render_out = renderer.render(
                rays_o,
                rays_d,
                near,
                far,
            )

            rgb_pred = render_out['rgb']
            gradient_error = render_out['gradient_error']
            weight_sum = render_out['weight_sum']

            # Compute Loss
            mask_sum = mask.sum() + 1e-5
            # RGB loss
            color_error = (rgb_pred - rgb_gt) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            # RGB metric
            psnr = compute_psnr(rgb_gt, rgb_pred, mask)

            # Eikonal loss
            eikonal_loss = gradient_error
            # Mask loss
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            # Sum all losses with weights
            loss = (
                color_fine_loss
                + eikonal_loss * config["eikonal_weight"]
                + mask_loss * config["mask_weight"]
            )
            # Compute gradients
            loss.backward()

            # Update network parameters (optimizer step)
            optimizer.step()

            # loss logging
            train_loss_running += loss.item()
            train_psnr_running += psnr.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                train_loss = train_loss_running / config["print_every_n"]
                train_psnr = train_psnr_running / config["print_every_n"]
                print(f'[{epoch:03d}/{iteration:06d}] train_loss: {train_loss:.6f} train_psnr: {train_psnr:.6f}')
                # Save best train model
                if train_loss < best_loss:
                    torch.save({
                        "sdf_field": sdf_field.state_dict(),
                        "color_field": color_field.state_dict(),
                        "variance_network": variance_network.state_dict(),
                    }, save_path / "model_best.ckpt")
                    best_loss = train_loss

                train_loss_running = 0.0
                train_psnr_running = 0.0

            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                sdf_field = sdf_field.eval()
                color_field = color_field.eval()
                variance_network = variance_network.eval()

                torch.cuda.empty_cache()
                try:
                    validate_images(renderer, val_dataloader.dataset, save_path, device=device)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('Could not validate because of CUDA out-of-memory. Skipping validation.')
                        torch.cuda.empty_cache()  # free memory again
                    else:
                        raise

                # Can change the voxel resolution to get a finer mesh
                vertices, faces = reconstruct_mesh(renderer, resolution=128)
                mesh = trimesh.Trimesh(vertices, faces)
                mesh.export(save_path / f"mesh.ply")

                sdf_field = sdf_field.train()
                color_field = color_field.train()
                variance_network = variance_network.train()

            # Save model checkpoint
            if iteration % config['save_every_n'] == (config['save_every_n'] - 1):
                torch.save({
                    "sdf_field": sdf_field.state_dict(),
                    "color_field": color_field.state_dict(),
                    "variance_network": variance_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, save_path / f"model.ckpt")

    print("Training end")


def validate_images(
    renderer: NeuSRenderer,
    val_dataset: DTUValDataset,
    save_dir: Path,
    batch_size=1024,
    device="cuda",
):
    """Render novel images, evaluate them, and save them to the save_dir.

    This function will be slow because we render the pixels in the batch.
    Change the batch_size to a smaller number if you encounter GPU memory issue.
    """

    print("Validating novel images...")
    total_psnr = 0.0
    for idx, batch in enumerate(val_dataset):
        rays_o, rays_d, rgb_gt, mask = batch
        h, w, _ = rays_o.shape

        rays_o = rays_o.view(-1, 3).to(device)
        rays_d = rays_d.view(-1, 3).to(device)
        rgb_gt = rgb_gt.view(-1, 3)
        mask = mask.view(-1, 1)

        near, far = DTUDataset.near_far_from_sphere(rays_o, rays_d)

        # Render all the pixels in batch to avoid memory issue
        collect_rgb_pixels = []
        for i in range(0, rays_o.shape[0], batch_size):
            render_out = renderer.render(
                rays_o[i:i + batch_size],
                rays_d[i:i + batch_size],
                near[i:i + batch_size],
                far[i:i + batch_size],
            )
            rgb = render_out['rgb'].detach().cpu()
            collect_rgb_pixels.append(rgb)
        rgb = torch.cat(collect_rgb_pixels, dim=0)
        psnr = compute_psnr(rgb_gt, rgb, mask)
        total_psnr += psnr.item()

        # Convert back to the shape (H, W, 3)
        rgb = rgb.reshape(h, w, 3)
        rgb_gt = rgb_gt.reshape(h, w, 3)
        image = (rgb * 255).clamp(0, 255).numpy().astype(np.uint8)
        image_gt = (rgb_gt * 255).clamp(0, 255).numpy().astype(np.uint8)
        image = Image.fromarray(np.concatenate([image_gt, image], axis=1))
        image.save(save_dir / f"{idx:04d}.png")

    print(f"Validation PSNR: {total_psnr / len(val_dataset)}")


def reconstruct_mesh(renderer: NeuSRenderer, resolution=64, threshold=0.0):
    # Assuming the object is within the bounding box [-1, 1]^3
    bound_min = torch.tensor([-1.01, -1.01, -1.01], dtype=torch.float32)
    bound_max = torch.tensor([1.01, 1.01, 1.01], dtype=torch.float32)
    sdf_grid = renderer.extract_fields(
        bound_min,
        bound_max,
        resolution=resolution,
        batch_size=4096,
    )
    sdf_grid = sdf_grid.numpy()
    # If you enounter that sdf_grid is all > 0, you might want to try train for longer and see
    vertices, faces, _, _ = marching_cubes(sdf_grid, level=threshold)
    # Rescale vertices to the original bounding box
    bound_min = bound_min.numpy()
    bound_max = bound_max.numpy()
    vertices = vertices / (resolution - 1) * (bound_max - bound_min) + bound_min
    return vertices, faces


def main(config):
    """
    Function for training NeuS model
    config: configuration dictionary with the following keys:
        'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
        'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
        'num_rays_per_image': number of rays per image
        'num_image_per_batch': number of images per batch
        'num_samples_init': number of initial sdf samples on a ray while training
        'num_samples_importance': number of importance sampling on a ray while training
        'eikonal_weight': weight of eikonal loss
        'mask_weight': weight of mask loss
        'learning_rate_model': learning rate of model optimizer
        'max_epochs': total number of epochs after which training should stop
        'print_every_n': print train loss every n iterations
        'validate_every_n': validate model every n iterations
        'save_every_n': save model checkpoint every n iterations
        'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Initialize train/val dataset and dataloader
    train_dataset = DTUTrainDataset(num_rays=config['num_rays_per_image'])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['num_image_per_batch'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = DTUValDataset()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"Train dataset size: {len(train_dataset)}. Val dataset size: {len(val_dataset)}")
    x = val_dataset[0]

    # Initialize model and optimizer
    sdf_field = SDFField().to(device)
    color_field = ColorField().to(device)
    variance_network = SingleVarianceNetwork().to(device)

    optimizer = torch.optim.Adam(
        list(sdf_field.parameters()) + list(color_field.parameters()) + list(variance_network.parameters()),
        lr=config['learning_rate_model'],
    )

    # load checkpoint if needed
    start_epoch = 0
    if config["resume_ckpt"] is not None:
        checkpoint = torch.load(config["resume_ckpt"])
        sdf_field.load_state_dict(checkpoint['sdf_field'])
        color_field.load_state_dict(checkpoint['color_field'])
        variance_network.load_state_dict(checkpoint['variance_network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    save_path = Path(f'exercise_3/runs/{config["experiment_name"]}')
    save_path.mkdir(exist_ok=True, parents=True)
    train(
        sdf_field,
        color_field,
        variance_network,
        optimizer,
        train_dataloader,
        val_dataloader,
        device,
        save_path,
        config,
        start_epoch=start_epoch,
    )


if __name__ == "__main__":
    config = {
        'experiment_name': 'neus',
        'device': 'cuda:0',
        'num_rays_per_image': 512,
        'num_image_per_batch': 2,
        'num_samples_init': 64,
        'num_samples_importance': 64,
        'eikonal_weight': 0.1,
        'mask_weight': 0.1,
        'learning_rate_model': 5e-4,
        'max_epochs': 800,
        'print_every_n': 50,
        'validate_every_n': 5000,
        'save_every_n': 1000,
        'resume_ckpt': None,
    }

    main(config)
