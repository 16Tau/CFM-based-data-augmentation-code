import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import glob


class PGDAttack:
    def __init__(self, model, eps=0.03, alpha=0.001, steps=10, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def generate_adversarial(self, image_tensor, target_class=None):
        adv_image = image_tensor.clone().detach()

        if self.random_start:
            delta = torch.rand_like(image_tensor) * 2 * self.eps - self.eps
            adv_image = torch.clamp(adv_image + delta, 0, 1)

        for step in range(self.steps):
            adv_image.requires_grad = True
            output = self.model(adv_image)

            if target_class is not None:
                loss = -nn.CrossEntropyLoss()(output, target_class)
            else:
                loss = nn.CrossEntropyLoss()(output, output.argmax(dim=1))

            self.model.zero_grad()
            loss.backward()

            grad = adv_image.grad.detach()
            adv_image = adv_image.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_image - image_tensor, -self.eps, self.eps)
            adv_image = torch.clamp(image_tensor + delta, 0, 1)

        return adv_image


def split_into_blocks(matrix, n):
    h, w = matrix.shape[:2]
    block_h = h // n
    block_w = w // n

    blocks = []
    for i in range(n):
        row_blocks = []
        for j in range(n):
            y_start = i * block_h
            y_end = (i + 1) * block_h if i < n - 1 else h
            x_start = j * block_w
            x_end = (j + 1) * block_w if j < n - 1 else w
            block = matrix[y_start:y_end, x_start:x_end]
            row_blocks.append(block)
        blocks.append(row_blocks)
    return blocks


def count_modified_pixels(original, modified):
    diff = np.abs(original.astype(np.float32) - modified.astype(np.float32))
    return np.sum(diff > 1e-6)


def apply_brighten_to_block(block, brightness_addition=50):
    if len(block.shape) == 3:
        result = np.zeros_like(block)
        for c in range(block.shape[2]):
            result[:, :, c] = np.clip(block[:, :, c].astype(np.int32) + brightness_addition, 0, 255).astype(np.uint8)
        return result
    else:
        return np.clip(block.astype(np.int32) + brightness_addition, 0, 255).astype(np.uint8)


def process_image_with_pgd(image_path, model, transform, eps, alpha, steps, thresholds, grid_size=9, sigma=2):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Failed to read image: {image_path}")
        return None, None

    image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    original_uint8 = image.copy()

    image_pil = Image.fromarray(original_uint8)
    image_tensor = transform(image_pil).unsqueeze(0)

    pgd_attack = PGDAttack(model, eps=eps, alpha=alpha, steps=steps)
    adv_tensor = pgd_attack.generate_adversarial(image_tensor)

    adv_image = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_image = np.clip(adv_image, 0, 1)
    adv_uint8 = (adv_image * 255).astype(np.uint8)

    return original_uint8, adv_uint8


def apply_block_brighten_based_on_modified_count(original, modified, thresholds, grid_size=9, brightness_addition=50, verbose=False):
    h, w = original.shape[:2]
    result = modified.copy()

    blocks = split_into_blocks(modified, grid_size)
    original_blocks = split_into_blocks(original, grid_size)

    modified_counts = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            count = count_modified_pixels(original_blocks[i][j], blocks[i][j])
            modified_counts[i, j] = count

    if verbose:
        print("\n=== Block tampering pixel count statistics ===")
        for i in range(grid_size):
            row_str = " ".join([f"{int(modified_counts[i, j]):4d}" for j in range(grid_size)])
        

    for i in range(grid_size):
        for j in range(grid_size):
            for threshold in thresholds:
                if modified_counts[i, j] < threshold:
                    y_start = i * (h // grid_size)
                    y_end = (i + 1) * (h // grid_size) if i < grid_size - 1 else h
                    x_start = j * (w // grid_size)
                    x_end = (j + 1) * (w // grid_size) if j < grid_size - 1 else w

                    block = result[y_start:y_end, x_start:x_end]
                    brightened_block = apply_brighten_to_block(block, brightness_addition)
                    result[y_start:y_end, x_start:x_end] = brightened_block
                    break

    return result, modified_counts


def main():
    input_dir = PATH_PLACEHOLDER
    output_dir = PATH_PLACEHOLDER

    os.makedirs(output_dir, exist_ok=True)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    eps = 0.03
    alpha = 0.001
    steps = 10
    thresholds = [1610]
    grid_size = 9
    brightness_addition = 255

    image_patterns = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_files:
        return


    for threshold in thresholds:
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        original, adv_image = process_image_with_pgd(
            img_path, model, transform, eps, alpha, steps, thresholds
        )
        if original is None:
            continue

        adv_output_path = os.path.join(output_dir, f"{base_name}_adv.png")
        cv2.imwrite(adv_output_path, adv_image[..., ::-1])

        first_threshold = thresholds[0]
        result, counts = apply_block_brighten_based_on_modified_count(
            original, adv_image, [first_threshold], grid_size, brightness_addition, verbose=True
        )
        threshold_dir = os.path.join(output_dir, f"threshold_{first_threshold}")
        output_path = os.path.join(threshold_dir, f"{base_name}_pgd_brightened.png")
        cv2.imwrite(output_path, result[..., ::-1])

        for threshold in thresholds[1:]:
            result, counts = apply_block_brighten_based_on_modified_count(
                original, adv_image, [threshold], grid_size, brightness_addition
            )
            threshold_dir = os.path.join(output_dir, f"threshold_{threshold}")
            output_path = os.path.join(threshold_dir, f"{base_name}_pgd_brightened.png")
            cv2.imwrite(output_path, result[..., ::-1])


if __name__ == '__main__':
    main()