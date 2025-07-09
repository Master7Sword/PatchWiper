import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def numerical_sort(value):
    return int(os.path.splitext(value)[0])


def get_dataset(name, train_dir, val_dir):
    if name == "PRWD":
        return PixabayDataset(train_dir), PixabayDataset(val_dir)
    elif name == "CLWD":
        return CLWDDataset(train_dir), CLWDDataset(val_dir)
    elif name == "ILAW":
        return ILAWDataset(train_dir), ILAWDataset(val_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    

transform = transforms.Compose([
    transforms.ToTensor()
])


class CLWDDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Flag to indicate whether the dataset is for training or testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = ['.jpg', '.png']


        self.groundtruth_dir = os.path.join(root_dir, 'Watermark_free_image')
        self.mask_dir = os.path.join(root_dir, 'Mask')
        self.watermarked_image_dir = os.path.join(root_dir, 'Watermarked_image')
        self.image_filenames = sorted(os.listdir(self.groundtruth_dir), key=numerical_sort)

    def __len__(self):
        return len(self.image_filenames)

    def _find_image_path(self, directory, filename):
        """
        Helper function to find the correct image path given a directory and filename.
        """
        for ext in self.extensions:
            path = os.path.join(directory, filename + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {filename} with supported extensions not found in {directory}")

    def __getitem__(self, idx):
        filename = os.path.splitext(self.image_filenames[idx])[0]  

        groundtruth_path = self._find_image_path(self.groundtruth_dir, filename)
        mask_path = self._find_image_path(self.mask_dir, filename)
        watermarked_image_path = self._find_image_path(self.watermarked_image_dir, filename)
        
        groundtruth = Image.open(groundtruth_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        watermarked_image = Image.open(watermarked_image_path).convert('RGB')
        
        if self.transform:
            groundtruth = self.transform(groundtruth)
            mask = self.transform(mask)
            watermarked_image = self.transform(watermarked_image)
            
        return {
            'groundtruth': groundtruth,
            'mask': mask,
            'watermarked_image': watermarked_image
        }



class ILAWDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Flag to indicate whether the dataset is for training or testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = ['.jpg', '.png']


        self.groundtruth_dir = os.path.join(root_dir, 'gt')
        self.alpha_dir = os.path.join(root_dir, 'alpha')
        self.watermarked_image_dir = os.path.join(root_dir, 'watermarked')
        self.image_filenames = sorted(os.listdir(self.groundtruth_dir), key=numerical_sort)

    def __len__(self):
        return len(self.image_filenames)

    def _find_image_path(self, directory, filename):
        """
        Helper function to find the correct image path given a directory and filename.
        """
        for ext in self.extensions:
            path = os.path.join(directory, filename + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {filename} with supported extensions not found in {directory}")

    def __getitem__(self, idx):
        filename = os.path.splitext(self.image_filenames[idx])[0]  

        groundtruth_path = self._find_image_path(self.groundtruth_dir, filename)
        alpha_path = self._find_image_path(self.alpha_dir, filename)
        watermarked_image_path = self._find_image_path(self.watermarked_image_dir, filename)
        
        groundtruth = Image.open(groundtruth_path).convert('RGB')
        alpha = Image.open(alpha_path).convert('L')
        mask = alpha.point(lambda p: 255 if p > 1 else 0) # 二值化
        watermarked_image = Image.open(watermarked_image_path).convert('RGB')
        
        if self.transform:
            groundtruth = self.transform(groundtruth)
            mask = self.transform(mask)
            watermarked_image = self.transform(watermarked_image)
            
        return {
            'groundtruth': groundtruth,
            'mask': mask,
            'watermarked_image': watermarked_image
        }



class PixabayDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Flag to indicate whether the dataset is for training or testing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = ['.jpg', '.png']


        self.groundtruth_dir = os.path.join(root_dir, 'Watermark_free_image')
        self.alpha_dir = os.path.join(root_dir, 'alpha')
        self.watermarked_image_dir = os.path.join(root_dir, 'Watermarked_image')
        self.image_filenames = sorted(os.listdir(self.groundtruth_dir), key=numerical_sort)

    def __len__(self):
        return len(self.image_filenames)

    def _find_image_path(self, directory, filename):
        """
        Helper function to find the correct image path given a directory and filename.
        """
        for ext in self.extensions:
            path = os.path.join(directory, filename + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {filename} with supported extensions not found in {directory}")

    def __getitem__(self, idx):
        filename = os.path.splitext(self.image_filenames[idx])[0]  

        groundtruth_path = self._find_image_path(self.groundtruth_dir, filename)
        alpha_path = self._find_image_path(self.alpha_dir, filename)
        watermarked_image_path = self._find_image_path(self.watermarked_image_dir, filename)
        
        groundtruth = Image.open(groundtruth_path).convert('RGB')
        alpha = Image.open(alpha_path).convert('L')
        mask = alpha.point(lambda p: 255 if p > 1 else 0) # 二值化
        watermarked_image = Image.open(watermarked_image_path).convert('RGB')
        
        if self.transform:
            groundtruth = self.transform(groundtruth)
            mask = self.transform(mask)
            watermarked_image = self.transform(watermarked_image)
            
        return {
            'groundtruth': groundtruth,
            'mask': mask,
            'watermarked_image': watermarked_image,
        }
    
    
class RealWorldDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = ['.jpg']

        self.watermarked_image_dir = os.path.join(root_dir, 'real-world-49')
        self.image_filenames = sorted(os.listdir(self.watermarked_image_dir), key=numerical_sort)

    def __len__(self):
        return len(self.image_filenames)

    def _find_image_path(self, directory, filename):
        """
        Helper function to find the correct image path given a directory and filename.
        """
        for ext in self.extensions:
            path = os.path.join(directory, filename + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {filename} with supported extensions not found in {directory}")

    def __getitem__(self, idx):
        filename = os.path.splitext(self.image_filenames[idx])[0]  
        watermarked_image_path = self._find_image_path(self.watermarked_image_dir, filename)
        watermarked_image = Image.open(watermarked_image_path).convert('RGB')
        
        if self.transform:
            watermarked_image = self.transform(watermarked_image)
            
        return {'watermarked_image': watermarked_image}