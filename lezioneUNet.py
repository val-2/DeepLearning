# cut_unet_with_perceptual_and_ssim.py

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image


# ----------------------------------------------------------
# 1) Data loading: Unpaired Dataset with trainA and trainB
# ----------------------------------------------------------

class UnpairedImageDataset(Dataset):
    """
    Loads images from two folders (trainA, trainB) without requiring pairing.
    Each __getitem__ returns one image from A, one image from B (randomly sampled).
    """
    def __init__(self, root_dir, phase="train", transform=None):
        super().__init__()
        self.dir_A = os.path.join(root_dir, phase + "A")
        self.dir_B = os.path.join(root_dir, phase + "B")

        self.A_paths = sorted(
            [os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        self.B_paths = sorted(
            [os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        self.len_A = len(self.A_paths)
        self.len_B = len(self.B_paths)
        self.transform = transform

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        # Cycle through A if idx >= len_A
        img_A = Image.open(self.A_paths[idx % self.len_A]).convert("RGB")
        # Randomly sample B
        img_B = Image.open(self.B_paths[random.randint(0, self.len_B - 1)]).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}


# ----------------------------------------------------------
# 2) Basic Blocks: Convolution, TransposedConv, etc.
# ----------------------------------------------------------

def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
               norm=True, activation="leaky"):
    """
    Standard conv block: Conv2d -> (InstanceNorm) -> Activation
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels, affine=False))
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        raise ValueError("Unsupported activation: {}".format(activation))
    return nn.Sequential(*layers)


def upconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm=True, activation="relu"):
    """
    Transposed conv block: ConvTranspose2d -> (InstanceNorm) -> Activation
    """
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                 bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels, affine=False))
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        raise ValueError("Unsupported activation: {}".format(activation))
    return nn.Sequential(*layers)


# ----------------------------------------------------------
# 3) U-Net Generator (Encoder-Decoder with skip connections)
# ----------------------------------------------------------

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(UNetGenerator, self).__init__()
        # Encoder: 8 layers down to 1×1 (for 256×256 input)
        self.enc1 = conv_block(in_channels, ngf, norm=False)       # 128×128
        self.enc2 = conv_block(ngf, ngf * 2)                       # 64×64
        self.enc3 = conv_block(ngf * 2, ngf * 4)                   # 32×32
        self.enc4 = conv_block(ngf * 4, ngf * 8)                   # 16×16
        self.enc5 = conv_block(ngf * 8, ngf * 8)                   # 8×8
        self.enc6 = conv_block(ngf * 8, ngf * 8)                   # 4×4
        self.enc7 = conv_block(ngf * 8, ngf * 8)                   # 2×2
        self.enc8 = conv_block(ngf * 8, ngf * 8, norm=False)       # 1×1

        # Decoder: 8 layers back to 256×256
        self.dec1 = upconv_block(ngf * 8, ngf * 8, norm=True)      # 2×2
        self.dec2 = upconv_block(ngf * 8 * 2, ngf * 8, norm=True)  # 4×4
        self.dec3 = upconv_block(ngf * 8 * 2, ngf * 8, norm=True)  # 8×8
        self.dec4 = upconv_block(ngf * 8 * 2, ngf * 8, norm=True)  # 16×16
        self.dec5 = upconv_block(ngf * 8 * 2, ngf * 4, norm=True)  # 32×32
        self.dec6 = upconv_block(ngf * 4 * 2, ngf * 2, norm=True)  # 64×64
        self.dec7 = upconv_block(ngf * 2 * 2, ngf, norm=True)      # 128×128
        self.dec8 = nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1)  # 256×256 (no norm, no act)
        self.tanh = nn.Tanh()

        # We’ll store intermediate feature maps for PatchNCE loss
        self.feature_layers = ["enc3", "enc4", "enc5", "enc6"]

    def forward(self, x, return_features=False):
        # Encoder forward
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder with skip connections
        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.dec8(d7)
        out = self.tanh(d8)

        if return_features:
            feats = {
                "enc3": e3,
                "enc4": e4,
                "enc5": e5,
                "enc6": e6
            }
            return out, feats
        else:
            return out


# ----------------------------------------------------------
# 4) PatchGAN Discriminator
# ----------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------
# 5) PatchNCE (Contrastive) Loss Module
# ----------------------------------------------------------

class PatchNCELoss(nn.Module):
    """
    Implements the PatchNCE (InfoNCE) loss as in CUT.
    """
    def __init__(self, nce_T=0.07, feat_dim=256, num_patches=256):
        super(PatchNCELoss, self).__init__()
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.cross_entropy = nn.CrossEntropyLoss()

    def _patch_correlation(self, feat_q, feat_k, MLP):
        B, C, H, W = feat_q.shape
        feat_q = feat_q.view(B, C, -1)   # [B, C, L]
        feat_k = feat_k.view(B, C, -1)   # [B, C, L]
        L = H * W

        idxs = torch.randperm(L, device=feat_q.device)[: self.num_patches]
        fq = feat_q[:, :, idxs]  # [B, C, N]
        fk = feat_k[:, :, idxs]  # [B, C, N]

        fq = fq.permute(0, 2, 1).reshape(-1, C)  # [B*N, C]
        fk = fk.permute(0, 2, 1).reshape(-1, C)

        q_proj = MLP(fq)  # [B*N, feat_dim]
        k_proj = MLP(fk)  # [B*N, feat_dim]
        q_norm = F.normalize(q_proj, dim=1)
        k_norm = F.normalize(k_proj, dim=1)

        logits = (q_norm @ k_norm.t()) / self.nce_T  # [B*N, B*N]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy(logits, labels)
        return loss

    def forward(self, feats_q, feats_k, MLPs):
        total_loss = 0.0
        for layer_name in feats_q.keys():
            total_loss += self._patch_correlation(
                feats_q[layer_name],
                feats_k[layer_name],
                MLPs[layer_name]
            )
        return total_loss / len(feats_q)


# ----------------------------------------------------------
# 6) Perceptual (VGG) Loss Module
# ----------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 pretrained on ImageNet.
    We extract features at:
      - relu1_2  (index: 3)
      - relu2_2  (index: 8)
      - relu3_2  (index: 17)
      - relu4_2  (index: 26)
    Then compute L1 distance between those feature maps.
    Input images are in [-1,1]. We convert to [0,1], then normalize with ImageNet stats.
    """
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
        # We only need layers up to 26 (relu4_2)
        self.slices = nn.ModuleDict({
            "relu1_2": vgg19[:4],     # conv1_1, relu1_1, conv1_2, relu1_2
            "relu2_2": vgg19[4:9],    # pool1, conv2_1, relu2_1, conv2_2, relu2_2
            "relu3_2": vgg19[9:18],   # pool2, conv3_1, relu3_1, conv3_2, relu3_2, ...
            "relu4_2": vgg19[18:27]   # pool3, conv4_1, relu4_1, conv4_2, relu4_2
        })
        for param in self.slices.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()
        self.device = device
        # ImageNet normalization after mapping [-1,1] → [0,1]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1].
        Return: sum of L1 distances between VGG feature maps at chosen layers.
        """
        # Convert to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0
        # Normalize
        gen_norm = (gen - self.mean) / self.std
        ref_norm = (ref - self.mean) / self.std

        loss = 0.0
        x_gen = gen_norm
        x_ref = ref_norm
        for key, slice_mod in self.slices.items():
            x_gen = slice_mod(x_gen)
            x_ref = slice_mod(x_ref)
            loss += self.l1(x_gen, x_ref)
        return loss


# ----------------------------------------------------------
# 7) SSIM Loss Module
# ----------------------------------------------------------

def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def create_window(window_size, channel, device):
    """
    Create a 2D Gaussian window to compute SSIM.
    window_size: e.g. 11
    channel: number of channels (3 for RGB)
    """
    _1D_window = gaussian(window_size, sigma=1.5).to(device).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # outer product → (window_size × window_size)
    window = _2D_window.unsqueeze(0).unsqueeze(0)  # shape (1,1,ws,ws)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss. Computes (1 - mean(SSIM map)) between two images.
    Assumes inputs are in [0,1].
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3  # assume 3 channels for RGB
        self.register_buffer("C1", torch.tensor((0.01 * 255) ** 2))
        self.register_buffer("C2", torch.tensor((0.03 * 255) ** 2))
        # The window will be created on first forward
        self.register_buffer("window", create_window(window_size, self.channel, device="cpu"))

    def _ssim_map(self, img1, img2, window, size_average=True):
        """
        Compute an SSIM map between img1 and img2 using a given window.
        img1, img2: [B, C, H, W], values in [0,1]
        window: [C,1,ws,ws]
        """
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1, C2 = self.C1, self.C2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1] or [0,1]. We'll assume [-1,1], so map to [0,1].
        Returns: 1 - mean(SSIM map over all pixels & channels).
        """
        # Convert from [-1,1] to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0

        # Ensure window is on same device
        window = self.window.to(gen.device)
        # Compute SSIM map
        if gen.shape[1] == self.channel:
            ssim_map = self._ssim_map(gen, ref, window, self.size_average)
        else:
            # If somehow not 3 channels, rebuild window for correct num channels
            channel = gen.shape[1]
            window = create_window(self.window_size, channel, gen.device)
            ssim_map = self._ssim_map(gen, ref, window, self.size_average)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            # Return a per-batch-item SSIM loss
            return 1 - ssim_map.view(ssim_map.size(0), -1).mean(1)


# ----------------------------------------------------------
# 8) Utility: Initialize weights (with NoneType guard)
# ----------------------------------------------------------

def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        # Initialize weights if they exist
        if hasattr(m, "weight") and m.weight is not None:
            if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in", nonlinearity="relu")
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
            elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, init_gain)
                    nn.init.constant_(m.bias.data, 0.0)

        # Initialize bias if it exists
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# ----------------------------------------------------------
# 9) Main Training Loop
# ----------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size), Image.BICUBIC),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Dataset & Dataloader
    dataset = UnpairedImageDataset(root_dir=args.dataroot,
                                   phase="train",
                                   transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            drop_last=True)

    # Initialize networks
    netG_A2B = UNetGenerator(in_channels=3,
                              out_channels=3,
                              ngf=args.ngf).to(device)
    netD_B = PatchGANDiscriminator(in_channels=3).to(device)

    # Initialize weights
    init_weights(netG_A2B, init_type=args.init_type)
    init_weights(netD_B, init_type=args.init_type)

    # Optimizers
    optimizer_G = optim.Adam(netG_A2B.parameters(),
                             lr=args.lr,
                             betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD_B.parameters(),
                             lr=args.lr,
                             betas=(0.5, 0.999))

    # Loss functions
    adversarial_criterion = nn.MSELoss().to(device)  # LSGAN
    patchnce_criterion = PatchNCELoss(nce_T=args.nce_T,
                                      feat_dim=args.nce_feat_dim,
                                      num_patches=args.num_patches).to(device)

    # Perceptual loss (optional)
    if args.use_perceptual:
        perceptual_criterion = VGGPerceptualLoss(device).to(device)

    # SSIM loss (optional)
    if args.use_ssim:
        ssim_criterion = SSIMLoss(window_size=11, size_average=True).to(device)

    # Create MLP projection heads for each selected layer
    MLPs = {}
    for layer_name in netG_A2B.feature_layers:
        # enc3: ngf*4, enc4: ngf*8, enc5: ngf*8, enc6: ngf*8
        if layer_name == "enc3":
            c_dim = args.ngf * 4
        else:
            c_dim = args.ngf * 8
        mlp = nn.Sequential(
            nn.Linear(c_dim, c_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_dim, args.nce_feat_dim, bias=False)
        ).to(device)
        init_weights(mlp, init_type=args.init_type)
        MLPs[layer_name] = mlp

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(1, args.n_epochs + 1):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            ###### Update Discriminator_B ######
            optimizer_D.zero_grad()
            pred_real = netD_B(real_B)
            loss_D_real = adversarial_criterion(
                pred_real,
                torch.full_like(pred_real, real_label, device=device)
            )
            fake_B = netG_A2B(real_A)  # return_features=False by default
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = adversarial_criterion(
                pred_fake,
                torch.full_like(pred_fake, fake_label, device=device)
            )
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            ###### Update Generator ######
            optimizer_G.zero_grad()
            fake_B, feats_fake = netG_A2B(real_A, return_features=True)
            pred_fake_for_G = netD_B(fake_B)
            loss_G_adv = adversarial_criterion(
                pred_fake_for_G,
                torch.full_like(pred_fake_for_G, real_label, device=device)
            )

            # Extract features from real_A for PatchNCE
            _, feats_real = netG_A2B(real_A, return_features=True)
            loss_NCE = patchnce_criterion(feats_fake, feats_real, MLPs)

            # Perceptual loss
            loss_percep = torch.tensor(0.0, device=device)
            if args.use_perceptual:
                loss_percep = perceptual_criterion(fake_B, real_A)

            # SSIM loss
            loss_ssim = torch.tensor(0.0, device=device)
            if args.use_ssim:
                loss_ssim = ssim_criterion(fake_B, real_A)

            # Total generator loss
            loss_G = (
                args.lambda_adv * loss_G_adv
                + args.lambda_nce * loss_NCE
                + args.lambda_perceptual * loss_percep
                + args.lambda_ssim * loss_ssim
            )
            loss_G.backward()
            optimizer_G.step()

            if (i + 1) % args.print_freq == 0:
                msg = (
                    f"[Epoch {epoch}/{args.n_epochs}] "
                    f"[Batch {i+1}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] "
                    f"[G adv: {loss_G_adv.item():.4f}, "
                    f"NCE: {loss_NCE.item():.4f}"
                )
                if args.use_perceptual:
                    msg += f", Percep: {loss_percep.item():.4f}"
                if args.use_ssim:
                    msg += f", SSIM: {loss_ssim.item():.4f}"
                msg += "]"
                print(msg)

        # Save sample images every epoch
        if epoch % args.save_epoch_freq == 0:
            with torch.no_grad():
                fake_B_sample, _ = netG_A2B(real_A, return_features=True)
                img_grid = torch.cat(
                    [real_A[:4], fake_B_sample[:4], real_B[:4]],
                    dim=0
                )
                save_image(
                    (img_grid + 1) / 2.0,
                    os.path.join(args.checkpoint_dir, f"epoch_{epoch}.png"),
                    nrow=4
                )

        # Save model checkpoints
        if epoch % args.checkpoint_freq == 0:
            torch.save(
                netG_A2B.state_dict(),
                os.path.join(args.checkpoint_dir, f"netG_A2B_epoch_{epoch}.pth")
            )
            torch.save(
                netD_B.state_dict(),
                os.path.join(args.checkpoint_dir, f"netD_B_epoch_{epoch}.pth")
            )
            for layer_name, mlp in MLPs.items():
                torch.save(
                    mlp.state_dict(),
                    os.path.join(
                        args.checkpoint_dir,
                        f"mlp_{layer_name}_epoch_{epoch}.pth"
                    )
                )


# ----------------------------------------------------------
# 10) Argument Parsing and Main
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and checkpoint paths
    parser.add_argument(
        "--dataroot", type=str, required=True,
        help="root directory of dataset containing trainA/ and trainB/"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="where to save models and images"
    )
    os.makedirs(parser.parse_known_args()[0].checkpoint_dir, exist_ok=True)

    # Model hyperparameters
    parser.add_argument(
        "--ngf", type=int, default=64,
        help="number of generator filters in first conv layer"
    )
    parser.add_argument(
        "--nce_feat_dim", type=int, default=256,
        help="feature dimension for PatchNCE MLP"
    )
    parser.add_argument(
        "--num_patches", type=int, default=256,
        help="number of patches to sample per layer for NCE loss"
    )
    parser.add_argument(
        "--nce_T", type=float, default=0.07,
        help="temperature for PatchNCE loss"
    )

    # Perceptual loss options
    parser.add_argument(
        "--use_perceptual", action="store_true",
        help="if set, add VGG-based perceptual loss between fake_B and real_A"
    )
    parser.add_argument(
        "--lambda_perceptual", type=float, default=0.0,
        help="weight for perceptual (VGG) loss"
    )

    # SSIM loss options
    parser.add_argument(
        "--use_ssim", action="store_true",
        help="if set, add SSIM loss between fake_B and real_A"
    )
    parser.add_argument(
        "--lambda_ssim", type=float, default=0.0,
        help="weight for SSIM loss"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="batch size"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=200,
        help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="learning rate"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5,
        help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=1.0,
        help="weight for adversarial loss"
    )
    parser.add_argument(
        "--lambda_nce", type=float, default=1.0,
        help="weight for PatchNCE loss"
    )

    # Image size and augmentation
    parser.add_argument(
        "--load_size", type=int, default=286,
        help="scale images to this size before cropping"
    )
    parser.add_argument(
        "--crop_size", type=int, default=256,
        help="size after random cropping"
    )

    # Misc
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="number of dataloader workers"
    )
    parser.add_argument(
        "--init_type", type=str, default="normal",
        choices=["normal", "xavier", "kaiming"],
        help="network initialization method"
    )
    parser.add_argument(
        "--print_freq", type=int, default=100,
        help="print losses every print_freq batches"
    )
    parser.add_argument(
        "--save_epoch_freq", type=int, default=1,
        help="save sample images every save_epoch_freq epochs"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10,
        help="save model checkpoints every checkpoint_freq epochs"
    )

    args = parser.parse_args()
    train(args)
