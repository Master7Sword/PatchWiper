import argparse
import os
import torch
import torch.nn as nn
from utils import *
from models.PatchWiper import PatchWiper
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description="Train PatchWiper with configurable parameters")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--device_ids', type=str, default="0,1,2,3")
    parser.add_argument('--dataset', type=str, choices=["PRWD", "CLWD", "ILAW"], default="PRWD")
    parser.add_argument('--dataset_dir', type=str, required=True, help="dataset root directory")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--schedule', type=int, default=50)
    parser.add_argument('--wln_ckpt', type=str, required=True, help="Watermark Localization Network checkpoint path")
    args = parser.parse_args()

    train_img_dir = os.path.join(args.dataset_dir, "train")
    val_img_dir = os.path.join(args.dataset_dir, "test")

    print(f"loading dataset {args.dataset}\n")
    training_set, val_set = get_dataset(args.dataset, train_img_dir, val_img_dir)
    train_loader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2)
    print("# of training samples: %d" % int(len(training_set)))
    print("# of validation samples: %d\n" % int(len(val_set)))

    # Build Model
    model = PatchWiper()
    print('model: #params={}'.format(compute_num_params(model, text=True)))
    model_structure(model)
    
    lpips_model = lpips.LPIPS(net='alex').cuda()

    # Parallel Training
    device_ids = [int(i) for i in args.device_ids.split(",")]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    # Load SegNet checkpoint
    WLN_ckpt_data = torch.load(args.wln_ckpt, weights_only=True)
    model.module.WLN.load_state_dict(WLN_ckpt_data)

    criterion = nn.L1Loss()
    criterion.cuda()

    # Only train RGN
    RGN_params = set(model.module.RGN.parameters())
    params_to_update = [p for p in model.parameters() if p in RGN_params]

    optimizer = torch.optim.Adam(params_to_update, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.schedule], gamma=0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.checkpoint_dir, f"tensorboard_logs/PatchWiper_{args.dataset}")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    best_psnr = 0.0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        PSNR_train = 0.0

        # Train
        tqdm_obj = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for i, data in enumerate(tqdm_obj):
            watermarked_image = data.get('watermarked_image').cuda()
            groundtruth = data.get('groundtruth').cuda()

            model.train()
            optimizer.zero_grad()

            out, _ = model(watermarked_image)
            out = torch.clamp(out, 0, 1)

            loss_fm = criterion(out, groundtruth)
            loss = loss_fm
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            psnr_train = batch_PSNR(out, groundtruth, 1.)
            # In case of inf
            if psnr_train > 50:
                psnr_train = 50
            PSNR_train += psnr_train
            tqdm_obj.set_postfix(loss=f"{loss.item():.4f}", PSNR_train=f"{psnr_train:.4f}")
        
        epoch_loss /= len(train_loader)
        PSNR_train /= len(train_loader)
        scheduler.step()

        # Validate
        psnr_val = 0
        ssim_val = 0
        rmse_val = 0
        rmsew_val = 0
        lpips_val = 0
        model.eval()
        with torch.no_grad():
            for _, val_data in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
                watermarked_image = val_data.get('watermarked_image').cuda()
                groundtruth = val_data.get('groundtruth').cuda()
                mask = val_data.get('mask').cuda()

                out_val = model(watermarked_image)
                out_val = torch.clamp(out_val, 0, 1)

                psnr_val_temp = batch_PSNR(out_val, groundtruth, 1.)
                if psnr_val_temp > 50:
                    psnr_val_temp = 50
                psnr_val += psnr_val_temp
                ssim_val += batch_SSIM(out_val, groundtruth, 1.)
                rmse_val += compute_RMSE(out_val, groundtruth, mask, is_w=False)
                rmsew_val += compute_RMSE(out_val, groundtruth, mask, is_w=True)
                lpips_val += batch_LPIPS(out_val, groundtruth, lpips_model)

        psnr_val /= len(val_loader)
        ssim_val /= len(val_loader)
        rmse_val /= len(val_loader)
        rmsew_val /= len(val_loader)
        lpips_val /= len(val_loader)
        tqdm.write(f"\n[epoch {epoch+1}] PSNR: {psnr_val:.4f} SSIM: {ssim_val:.4f} RMSE: {rmse_val:.4f} RMSEw: {rmsew_val:.4f} LPIPS: {lpips_val:.4f}\n")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('PSNR/train', PSNR_train, epoch)
        writer.add_scalar('PSNR/val', psnr_val, epoch)
        writer.add_scalar('SSIM/val', ssim_val, epoch)
        writer.add_scalar('RMSE/val', rmse_val, epoch)
        writer.add_scalar('RMSEw/val', rmsew_val, epoch)

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            model_path = os.path.join(args.checkpoint_dir, f"PatchWiper({args.dataset}).pth")
            torch.save(model.module.state_dict(), model_path)
            print("saved best model with psnr: ", best_psnr)

    writer.close()

if __name__ == "__main__":
    main()

