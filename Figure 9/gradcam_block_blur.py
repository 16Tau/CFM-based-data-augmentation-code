import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import glob


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, class_idx=None):
        output = self.model(input_image)

        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        return cam


def process_image_with_gradcam(image_path, model, transform, target_layer, thresholds, grid_size=9):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Failed to read image: {image_path}")
        return None

    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    image_uint8 = (image * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8)

    image_tensor = transform(image_pil).unsqueeze(0)

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(image_tensor)

    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

    return image, cam_resized


def apply_block_brighten(image, cam, thresholds, grid_size=9, brightness_addition=0.3):
    h, w = image.shape[:2]
    block_h = h // grid_size
    block_w = w // grid_size

    result_image = image.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * block_h
            y_end = (i + 1) * block_h if i < grid_size - 1 else h
            x_start = j * block_w
            x_end = (j + 1) * block_w if j < grid_size - 1 else w

            block_cam = cam[y_start:y_end, x_start:x_end]
            cam_mean = np.mean(block_cam)

            for threshold in thresholds:
                if cam_mean < threshold:
                    block = result_image[y_start:y_end, x_start:x_end]
                    brightened_block = np.clip(block + brightness_addition, 0, 1)
                    result_image[y_start:y_end, x_start:x_end] = brightened_block
                    break

    return result_image


def main():
    input_dir = PATH_PLACEHOLDER
    output_dir = PATH_PLACEHOLDER

    os.makedirs(output_dir, exist_ok=True)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    target_layer = model.layer4[-1]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    thresholds = [0.1,0.2, 0.35, 0.45, 0.25, 0.65]

    image_patterns = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_files:
        return


    for threshold in thresholds:
        threshold_dir = os.path.join(output_dir, f"threshold_{int(threshold * 100)}")
        os.makedirs(threshold_dir, exist_ok=True)

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        image, cam = process_image_with_gradcam(img_path, model, transform, target_layer, thresholds)
        if image is None:
            continue

        cam_vis = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_vis = cv2.cvtColor(cam_vis, cv2.COLOR_BGR2RGB) / 255.0
        cam_output_path = os.path.join(output_dir, f"{base_name}_cam.png")
        cv2.imwrite(cam_output_path, (cam_vis * 255).astype(np.uint8)[..., ::-1])

        for threshold in thresholds:
            result = apply_block_brighten(image.copy(), cam, [threshold], grid_size=9, brightness_addition=100)
            threshold_dir = os.path.join(output_dir, f"threshold_{int(threshold * 100)}")
            output_path = os.path.join(threshold_dir, f"{base_name}_brightened.png")
            cv2.imwrite(output_path, (result * 255).astype(np.uint8)[..., ::-1])


if __name__ == '__main__':
    main()