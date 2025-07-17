import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.WLN import WatermarkLocalizationNetwork 
from models.WLNModules import IOU
from utils import *
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description="Train SegNet with configurable parameters")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--device_ids', type=str, default="0,1")
    parser.add_argument('--dataset', type=str, choices=["PRWD", "CLWD", "ILAW"], default="PRWD")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint", help="checkpoint directory")
    parser.add_argument('--dataset_dir', type=str, required=True, help="dataset directory")
    parser.add_argument('--lambda_iou', type=float, default=0.25)
    parser.add_argument('--lambda_primary', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=8, help="number of threads to load dataset")
    parser.add_argument('--schedule', type=int, default=65, help="Epoch to reduce learning rate")
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
    model = WatermarkLocalizationNetwork()
    print('model: #params={}'.format(compute_num_params(model, text=True)))

    # Parallel Training
    device_ids = [int(i) for i in args.device_ids.split(",")]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    bce_loss = nn.BCELoss()
    iou_loss = IOU(size_average=True)

    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.schedule], gamma=0.1)

    tb_log_dir = os.path.join(args.checkpoint_dir, f"tensorboard_logs/SegNet_{args.dataset}")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        # Train
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")):
            watermarked_image = data.get('watermarked_image').cuda()
            mask = data.get('mask').cuda()

            model.train()
            optimizer.zero_grad()

            output_masks = model(watermarked_image)
            pred_ms = [F.interpolate(ms, size=mask.shape[2:], mode='bilinear') for ms in output_masks]
            pred_ms = [pred_m.clamp(0,1) for pred_m in pred_ms]
            final_mask_loss = bce_loss(pred_ms[0], mask)
            primary_mask = pred_ms[1::2][::-1]
            self_calibrated_mask = pred_ms[2::2][::-1]
            primary_loss =  sum([bce_loss(pred_m, mask) * (args.gamma**i) for i,pred_m in enumerate(primary_mask)])
            self_calibrated_loss =  sum([bce_loss(pred_m, mask) * (args.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)])
            if args.lambda_iou > 0:
                self_calibrated_loss += sum([iou_loss(pred_m, mask) * (args.gamma**i) for i,pred_m in enumerate(self_calibrated_mask)]) * args.lambda_iou

            loss = final_mask_loss + self_calibrated_loss + args.lambda_primary * primary_loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        epoch_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_metrics_sum = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'iou': 0.0,
            'f1_score': 0.0
        }
        with torch.no_grad():
            for i, val_data in enumerate(tqdm(val_loader, desc="Validation", unit="batch")):
                watermarked_image = val_data.get('watermarked_image').cuda()
                mask = val_data.get('mask').cuda()

                out_val = model(watermarked_image)[0]
                predicted_masks = (out_val > 0.5).float()

                metrics = calculate_metrics(predicted_masks.cpu().numpy(), mask.cpu().numpy())
                for key in val_metrics_sum:
                    val_metrics_sum[key] += metrics[key]

        avg_val_metrics = {key: val_metrics_sum[key] / len(val_loader) for key in val_metrics_sum}
        tqdm.write(f"\nacc: {avg_val_metrics['accuracy']:.4f} precision: {avg_val_metrics['precision']:.4f} recall: {avg_val_metrics['recall']:.4f} iou: {avg_val_metrics['iou']:.4f} F1: {avg_val_metrics['f1_score']:.4f}\n")
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Val/accuracy', avg_val_metrics['accuracy'], epoch)
        writer.add_scalar('Val/precision', avg_val_metrics['precision'], epoch)
        writer.add_scalar('Val/recall', avg_val_metrics['recall'], epoch)
        writer.add_scalar('Val/iou', avg_val_metrics['iou'], epoch)
        writer.add_scalar('Val/f1_score', avg_val_metrics['f1_score'], epoch)

        if avg_val_metrics['accuracy'] > best_acc:
            best_acc = avg_val_metrics['accuracy']
            model_path = os.path.join(args.checkpoint_dir, f"WLN({args.dataset.upper()}).pth")
            torch.save(model.module.state_dict(), model_path)
            print("saved best model with acc: ", best_acc)

    writer.close()

if __name__ == "__main__":
    main()