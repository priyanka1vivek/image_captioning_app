import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

images_path = r"C:\Users\Swetha\OneDrive\Desktop\image_captioning\val2017"
captions_path = r"C:\Users\Swetha\OneDrive\Desktop\image_captioning\annotations_trainval2017\captions_val2017.json"

# custom dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, images_path, captions_file, transform=None):
        self.images_path = images_path
        self.transform = transform

        # loading JSON captions
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)

        #  image ID to file name mapping
        self.id_to_filename = {img['id']: img['file_name'] for img in captions_data['images']}

        # extract annotations (image_id -> caption)
        self.image_caption_pairs = []
        for ann in captions_data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']                
            filename = self.id_to_filename[img_id]
            self.image_caption_pairs.append((filename, caption))

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        filename, caption = self.image_caption_pairs[idx]
        img_path = os.path.join(self.images_path, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

# image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# create dataset
dataset = ImageCaptionDataset(images_path, captions_path, transform=transform)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test loading
for imgs, caps in data_loader:
    print("Image batch shape:", imgs.shape)
    print("Captions:", caps[:3])
    break
