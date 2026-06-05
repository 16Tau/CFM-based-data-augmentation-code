import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import num_classes
from train_wth_val import MultiRootFolderDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx=None):
        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()

        class_score = output[:, class_idx]
        class_score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))

        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        return cam

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MultiRootFolderDataset(root_folders=[PATH_PLACEHOLDER], transform=transform)

weights_dir = PATH_PLACEHOLDER
weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]

output_base = PATH_PLACEHOLDER
os.makedirs(output_base, exist_ok=True)

def get_superimposed_image(img, cam):
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed

def save_superimposed_image(img, cam, save_path):
    superimposed = get_superimposed_image(img, cam)
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_by_class(per_class_count=100):
    class_images = {i: [] for i in range(len(train_dataset.class_to_idx))}

    for i, (img_path, label) in enumerate(zip(train_dataset.image_paths, train_dataset.labels)):
        if len(class_images[label]) < per_class_count:
            class_images[label].append((i, img_path, label))

    all_image_info = []
    for class_idx, img_list in class_images.items():
        all_image_info.extend(img_list)

    for weight_file in weight_files:
        print(f"Processing {weight_file}...")

        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        weight_path = os.path.join(weights_dir, weight_file)
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()

        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)

        weight_name = os.path.splitext(weight_file)[0]
        output_dir = os.path.join(output_base, weight_name)
        os.makedirs(output_dir, exist_ok=True)

        total_images = len(all_image_info)
        processed = 0

        for img_index, img_path, label in all_image_info:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_tensor = transform(Image.fromarray(img)).unsqueeze(0)
            cam = grad_cam.generate_cam(img_tensor)

            save_path = os.path.join(output_dir, f'{img_index:05d}.png')
            save_superimposed_image(img, cam, save_path)

            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed}/{total_images} images...")

        print(f"Completed {weight_file}")

    return all_image_info

def create_combined_images(all_image_info):
    num_weights = len(weight_files)
    num_cols = min(4, num_weights)
    num_rows = (num_weights + num_cols - 1) // num_cols

    weight_names = [os.path.splitext(w)[0] for w in weight_files]

    for img_index, img_path, label in all_image_info:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        if num_rows == 1 and num_cols > 1:
            axes = axes.reshape(1, -1)
        elif num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])

        for idx, wf in enumerate(weight_files):
            wname = weight_names[idx]
            img_path_check = os.path.join(output_base, wname, f'{img_index:05d}.png')
            if os.path.exists(img_path_check):
                img = cv2.imread(img_path_check)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                row = idx // num_cols
                col = idx % num_cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(wname, fontsize=8)
                    axes[row, col].axis('off')
            else:
                row = idx // num_cols
                col = idx % num_cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].axis('off')

        for idx in range(len(weight_files), axes.shape[0] * axes.shape[1]):
            row = idx // num_cols
            col = idx % num_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_base, f'{img_index:05d}_combined.png')
        plt.savefig(save_path)
        plt.close()

        if img_index % 500 == 0:
            print(f"Saved combined image {img_index}/{len(all_image_info)}")

if __name__ == '__main__':
    per_class_count = 100
    print(f"Processing {per_class_count} images per class...")
    print(f"Total classes: {len(train_dataset.class_to_idx)}")
    print(f"Total images to process: {per_class_count * len(train_dataset.class_to_idx)}")
    all_image_info = process_by_class(per_class_count)

    print("\nCreating combined images for each original image...")
    create_combined_images(all_image_info)

    print("All images created!")
