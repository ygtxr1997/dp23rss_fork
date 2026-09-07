import torch
import torchvision

from tqdm import tqdm

from diffusion_policy.model.roi.jepa_helper import JEPAHelper


def de_jepa():
    """
    Usage:
    export TORCH_HOME="/home/geyuan/pretrained/torch"
    """

    # preprocessor
    processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')

    frames_T_H_W_C, _, info = torchvision.io.read_video(
      "/home/geyuan/code/dp23rss_fork/debug/libero384_roi.mp4",
      pts_unit="sec",
    )  # uint8, [0,255]
    print("Reading frames:", frames_T_H_W_C.shape, frames_T_H_W_C.min(), frames_T_H_W_C.max())

    x_C_T_H_W = processor(frames_T_H_W_C.permute(0, 3, 1, 2))[0]  # float32, [-3,3]
    print(type(processor.normalize))
    print((processor.normalize))
    print("Processed:", x_C_T_H_W.shape, x_C_T_H_W.dtype, x_C_T_H_W.min(), x_C_T_H_W.max(), x_C_T_H_W.mean())

    # models
    # V-JEPA 2
    # vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
    # vjepa2_vit_huge = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
    # vjepa2_vit_giant = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
    # vjepa2_vit_giant_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')
    # V-JEPA 2.1
    # vjepa2_1_vit_base_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')
    # vjepa2_1_vit_large_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
    vjepa2_1_vit_large_384 = torch.hub.load(
        '/home/geyuan/pretrained/torch/hub/facebookresearch_vjepa2_main',
        'vjepa2_1_vit_large_384',
        source="local",
    )
    # vjepa2_1_vit_giant_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_giant_384')
    # vjepa2_1_vit_gigantic_384 = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')

    jepa_model = vjepa2_1_vit_large_384

    print("tuple length:", len(jepa_model))
    for i, item in enumerate(jepa_model):
      print(i, type(item))
      if hasattr(item, "forward"):
          print("  parameters:", f"{sum(p.numel() for p in item.parameters()) / 1024 / 1024 / 1024:.2f}B")

    net = jepa_model[0].cuda()

    x = torch.randn(4, 3, 16, 384, 384).cuda()
    z = net(x)
    print("z.shape:", z.shape, z.min(), z.max(), z.mean())

    z_processed = net(x_C_T_H_W.unsqueeze(0).cuda())
    print("z_processed.shape:", z_processed.shape, z_processed.min(), z_processed.max(), z_processed.mean())
    """
    [B, T/2 × H/16 × W/16, 1024]
    z_processed.shape: torch.Size([1, 25344, 1024]) tensor(-32.7461, device='cuda:0', grad_fn=<MinBackward1>) 
    tensor(33.0037, device='cuda:0', grad_fn=<MaxBackward1>) tensor(-0.0138, device='cuda:0', grad_fn=<MeanBackward0>)
    """

    print("OK!")


def de_jepahelper():
    """
    Usage:
    export PYTHONPATH=~/code/dp23rss_fork/
    """
    jepa_helper = JEPAHelper(
        model_name="vjepa2_1_vit_base_384",
        torch_home="/home/geyuan/pretrained/torch",
        device="cuda",
    )

    frames_T_H_W_C, _, info = torchvision.io.read_video(
        "/home/geyuan/code/dp23rss_fork/debug/libero384_roi.mp4",
        pts_unit="sec",
    )  # uint8, [0,255]
    frames_T_H_W_C = frames_T_H_W_C[:1]
    frames_B_C_T_H_W = frames_T_H_W_C.permute(3, 0, 1, 2).unsqueeze(0).to(device='cuda')

    batch_size = 1
    frames_B_C_T_H_W = frames_B_C_T_H_W.repeat(batch_size, 1, 1, 1, 1)
    frames_B_C_T_H_W = torch.randn(batch_size, 3, 1, 256, 256).cuda()
    print("Input:", frames_B_C_T_H_W.shape)

    for _ in tqdm(range(100)):
        with torch.no_grad():
            latent_B_N_D = jepa_helper.encode(frames_B_C_T_H_W)
    print("latent_B_N_D:", latent_B_N_D.shape, latent_B_N_D.min(), latent_B_N_D.max(), latent_B_N_D.mean())

    print("OK!")


if __name__ == "__main__":
    # de_jepa()
    de_jepahelper()