import os
import torch
import glob
import argparse
from PIL import Image
import numpy as np
from models.PatchWiper import PatchWiper


def save_image(image_array, file_path, mode):
    image_array = (image_array * 255).astype(np.uint8)

    if image_array.shape[-1] == 1:
        image_array = image_array.squeeze(-1)

    if mode == "RGB":
        image = Image.fromarray(image_array, mode='RGB')
    elif mode == "L":
        image = Image.fromarray(image_array, mode='L')
    else:
        print("Error, mode must be RGB or L")
        exit()
    image.save(file_path)


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img


def process_images(model, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, '*.*'))
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [path for path in image_paths
                  if os.path.splitext(path)[1].lower() in image_extensions]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    model.eval()
    
    for image_path in image_paths:
        try:
            img_name = os.path.basename(image_path)
            print(f"Processing {img_name}...")
            img = load_image(image_path).to(device)

            with torch.no_grad():
                bg, mask = model.mask_prediction(img)
                bg = torch.clamp(bg, 0, 1)

            # Save output
            base_name = os.path.splitext(img_name)[0] 
            bg_output_path = os.path.join(output_dir, f"{base_name}_bg.jpg")  
            mask_output_path = os.path.join(output_dir, f"{base_name}_mask.jpg")  
            bg_output_np = bg.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_output_np = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            save_image(bg_output_np, bg_output_path, mode="RGB")
            save_image(mask_output_np, mask_output_path, mode="L")
            print(f"Saved result for {img_name}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Test PatchWiper with configurable parameters")
    parser.add_argument('--input_dir', type=str, default="/custom", help="Test image input directory")
    parser.add_argument('--output_dir', type=str, default="/custom/output", help="Test image output directory")
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PatchWiper().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, weights_only=True))

    process_images(model, input_dir, output_dir, device)


if __name__ == '__main__':
    main()