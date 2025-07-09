import argparse
from utils import *
from models.PatchWiper import PatchWiper
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Test PatchWiper with configurable parameters")
    parser.add_argument('--dataset', type=str, choices=["PRWD", "CLWD", "ILAW"], default="PRWD")
    parser.add_argument('--dataset_dir', type=str, default=None, help="Test image directory")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device_ids', type=str, default="0,1")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    train_img_dir = os.path.join(args.dataset_dir, "train")
    test_img_dir = os.path.join(args.dataset_dir, "test")

    print(f"loading dataset {args.dataset}\n")
    _, val_set = get_dataset(args.dataset, train_img_dir, test_img_dir)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)
    print("# of test samples: %d\n" % int(len(val_set)))

    # Build Model
    model = PatchWiper()
    print('model: #params={}'.format(compute_num_params(model, text=True)))
    model_structure(model)

    lpips_model = lpips.LPIPS(net='alex').cuda()

    # Parallel Test
    device_ids = [int(i) for i in args.device_ids.split(",")]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    model.module.load_state_dict(torch.load(args.ckpt_path, weights_only=True))

    # Validate
    model.eval()
    psnr_val = 0
    ssim_val = 0
    rmse_val = 0
    rmsew_val = 0
    lpips_val = 0
    val_metrics_sum = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'iou': 0.0,
        'f1_score': 0.0
    }
    with torch.no_grad():
        for _, test_data in enumerate(tqdm(test_loader, desc="Test", unit="batch")):
            watermarked_image = test_data.get('watermarked_image').cuda()
            groundtruth = test_data.get('groundtruth').cuda()
            mask = test_data.get('mask').cuda()

            out_bg, out_mask = model(watermarked_image)
            out_bg = torch.clamp(out_bg, 0, 1)
            out_mask = (out_mask > 0.5).float()

            psnr_val_temp = batch_PSNR(out_bg, groundtruth, 1.)
            if psnr_val_temp > 50:
                psnr_val_temp = 50
            psnr_val += psnr_val_temp
            ssim_val += batch_SSIM(out_bg, groundtruth, 1.)
            rmse_val += compute_RMSE(out_bg, groundtruth, mask, is_w=False)
            rmsew_val += compute_RMSE(out_bg, groundtruth, mask, is_w=True)
            lpips_val += batch_LPIPS(out_bg, groundtruth, lpips_model)

            metrics = calculate_metrics(out_mask.cpu().numpy(), mask.cpu().numpy())
            for key in val_metrics_sum:
                val_metrics_sum[key] += metrics[key]

    psnr_val /= len(test_loader)
    ssim_val /= len(test_loader)
    rmse_val /= len(test_loader)
    rmsew_val /= len(test_loader)
    lpips_val /= len(test_loader)
    avg_val_metrics = {key: val_metrics_sum[key] / len(test_loader) for key in val_metrics_sum}
    
    tqdm.write(f"\nPSNR: {psnr_val:.4f} SSIM: {ssim_val:.4f} RMSE: {rmse_val:.4f} RMSEw: {rmsew_val:.4f} LPIPS: {lpips_val:.4f}\n")
    tqdm.write(f"\nacc: {avg_val_metrics['accuracy']:.4f} precision: {avg_val_metrics['precision']:.4f} recall: {avg_val_metrics['recall']:.4f} iou: {avg_val_metrics['iou']:.4f} F1: {avg_val_metrics['f1_score']:.4f}\n")


if __name__ == "__main__":
    main()
