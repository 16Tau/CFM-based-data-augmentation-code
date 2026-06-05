import random

import torch
import torch.nn as nn
from PIL import Image
import json
import torchvision.transforms as transforms
from attack import Attack
import numpy as np
import z3
from feature_classifier.constants import *
from scipy.ndimage import gaussian_filter

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

channel_clamps = {
    0: {"min": -2.1179, "max": 2.2489},  # channel 0 (R)    1: {"min": -2.0357, "max": 2.4285},  # channel 1 (G)    2: {"min": -1.8044, "max": 2.64}  # channel 2 (B)}


def map_to_224x224(input_matrix, n, output_channels=3):
    """
    Map n x n matrix to 224 x 224 (single channel) or 3 x 224 x 224 (three channels) matrix.    Parameters:
        input_matrix: Input matrix with shape (n, n)
        n: Dimension of input matrix        output_channels: Number of output channels (1 or 3)    Returns:
        output_matrix: Matrix with shape (output_channels, 224, 224)    """
    assert input_matrix.shape == (n, n), f"Input matrix shape must be({n}, {n})"

    # # Initialize output matrix    if output_channels == 1:
        output_matrix = np.zeros((224, 224), dtype=np.float32)
    else:
        output_matrix = np.zeros((output_channels, 224, 224), dtype=np.float32)

    # # Use np.array_split indices for non-uniform division
    h_splits = np.array_split(range(224), n)
    w_splits = np.array_split(range(224), n)

    for i in range(n):
        for j in range(n):
            # # Calculate index range for current block            h_start, h_end = h_splits[i][0], h_splits[i][-1] + 1
            w_start, w_end = w_splits[j][0], w_splits[j][-1] + 1
            # # Fill corresponding block with input_matrix[i, j] value
            if output_channels == 1:
                output_matrix[h_start:h_end, w_start:w_end] = input_matrix[i, j]
            else:
                for c in range(output_channels):
                    output_matrix[c, h_start:h_end, w_start:w_end] = input_matrix[i, j]

    return output_matrix


def split_into_blocks(matrix, n):
    """
    Non-uniformly divide matrix into n x n blocks.    """
    assert matrix.shape == (3, 224, 224), "Input matrix shape must be(3, 224, 224)"

    # Divide height and width
    h_splits = np.array_split(range(224), n)
    w_splits = np.array_split(range(224), n)

    blocks = []
    for h in range(n):
        row_blocks = []
        for w in range(n):
            h_start, h_end = h_splits[h][0], h_splits[h][-1] + 1
            w_start, w_end = w_splits[w][0], w_splits[w][-1] + 1
            block = matrix[:, h_start:h_end, w_start:w_end]
            row_blocks.append(block)
        blocks.append(row_blocks)

    return blocks


def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()

    return predicted_class_index, output[index].item()


def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):
    adv_image = actual_image_matrix.copy()

    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.detach().cpu().numpy()

    factors = np.array([[0.229, 0.485], [0.224, 0.456], [0.225, 0.406]])
    scales = np.array([255, 255, 255])

    for c in range(3):
        actual_transform = actual_image_transform_matrix[c]
        adversarial_transform = adversarial_sample_transform_matrix[c]
        actual_image = actual_image_matrix[c]

        mask_greater = adversarial_transform > actual_transform
        mask_less = adversarial_transform < actual_transform
        mask_equal = adversarial_transform == actual_transform

        adv_image[c] = np.where(mask_greater,
                                np.ceil((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                np.where(mask_less,
                                         np.floor((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                         np.where(mask_equal, actual_image, adv_image[c])))

    adv_image = np.clip(adv_image, 0, 255)

    return adv_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CFM(Attack):
    def __init__(self, model, eps=float(np.sqrt((8 ** 2) * 3 * 224 * 224)), alpha=1 / 255, steps=10, random_start=True,
                 vcr=0.9, mask=np.zeros((3, 224, 224), dtype=np.float32)):
        super().__init__("IFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.vcr = vcr
        self.mask = mask

    def forward(self, images, labels, labels_top2, actual_image_transform_matrix, actual_image_matrix, actual_image):

        model_current = model_type

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels_top2 = labels_top2.clone().detach().to(self.device)

        images.requires_grad = True

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        mask = torch.from_numpy(self.mask).to(device)

        specific_values = [-0.017, 0,0.017]
        specific_values = [-0.1, 0,0.1]
        adv_image_transform_matrix = actual_image_transform_matrix.copy()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost_top2 = loss(outputs, labels_top2)

            # # First gradient calculation            outputs = self.get_logits(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            # Recalculate output to generate new computation graph
            outputs = self.get_logits(adv_images)
            cost_top2 = loss(outputs, labels_top2)
            grad_top2 = torch.autograd.grad(
                cost_top2, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = grad.squeeze(0).cpu()
            grad_np = grad.cpu().numpy()
            grad_top2 = grad_top2.squeeze(0).cpu()
            grad_top2_np = grad_top2.cpu().numpy()

            splited_grad = split_into_blocks(grad_np, parts)
            splited_grad_max = np.zeros((parts, parts))
            splited_top2_grad = split_into_blocks(grad_top2_np, parts)
            splited_top2_grad_max = np.zeros((parts, parts))
            for i in range(parts):
                for j in range(parts):
                    splited_grad_max[i][j] = np.average(splited_grad[i][j])
                    splited_top2_grad_max[i][j] = np.average(splited_top2_grad[i][j])

            # # Use Z3 optimizer            solver = z3.Optimize()
            variables = [[0 for j in range(parts)] for i in range(parts)]

            # Define absolute value sum
            abs_sum = z3.Sum([z3.Abs(variables[i][j]) for i in range(parts) for j in range(parts)])

            # solver.reset()
            for i in range(parts):
                for j in range(parts):
                    var = z3.Real(f'x_{i}_{j}')
                    variables[i][j] = var
                    value_constraints = [var == val for val in specific_values]
                    solver.add(z3.Or(value_constraints))
                    if splited_grad_max[i][j] != splited_top2_grad_max[i][j]:  # # Avoid 0 > 0 contradiction                        solver.add(var * splited_grad_max[i][j] > var * splited_top2_grad_max[i][j])

            # Add minimization objective
            solver.minimize(abs_sum)

            check_result = solver.check()
            if check_result == z3.sat:
                model = solver.model()
                result_matrix = np.zeros(shape=(parts, parts))
                for i in range(parts):
                    for j in range(parts):
                        var = variables[i][j]
                        if isinstance(var, z3.ExprRef):
                            var_value = float(model[var].as_decimal(prec=10).rstrip('?'))
                        else:
                            var_value = 0.0
                        # if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                        result_matrix[i][j] = var_value

                # # Map to 224x224
                mapped_var = map_to_224x224(result_matrix, parts, 1)
                adv_image_transform_matrix = adv_image_transform_matrix.copy()

                # # Modified part: Calculate boundaries for non-uniform block division
                h_splits = np.array_split(range(224), parts)
                w_splits = np.array_split(range(224), parts)

                # # Gaussian blur parameters: sigma can be adjusted as needed (set to 1.0 here for moderate blur strength)
                sigma = 2.0

                # # Block-level processing: Apply Gaussian blur to blocks where result_matrix[i][j] > 0                for blk_i in range(parts):
                    for blk_j in range(parts):
                        if result_matrix[blk_i][blk_j] > 0:
                            # # Get current block boundaries
                            h_start, h_end = h_splits[blk_i][0], h_splits[blk_i][-1] + 1
                            w_start, w_end = w_splits[blk_j][0], w_splits[blk_j][-1] + 1

                            # # Apply Gaussian blur to each channel
                            for k in range(3):
                                # # Extract current block region                                block_region = adv_image_transform_matrix[k, h_start:h_end, w_start:w_end].copy()

                                # # Apply Gaussian blur (order=0 for 2D blur)                                blurred_block = gaussian_filter(block_region, sigma=sigma, order=0)

                                # # Put blurred block back to original matrix                                adv_image_transform_matrix[k, h_start:h_end, w_start:w_end] = blurred_block

                m = (adv_image_transform_matrix == actual_image_transform_matrix)

                for ch in range(3):
                    ch_min = channel_clamps[ch]["min"]
                    ch_max = channel_clamps[ch]["max"]
                    adv_image_transform_matrix[ch] = np.clip(adv_image_transform_matrix[ch], ch_min, ch_max)

                adv_images = torch.tensor(
                    adv_image_transform_matrix,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                adv_images.requires_grad = True
            elif check_result == z3.unsat:
                break
            else:
                break

            delta = adv_images - images

            # Tampering intensity is 32
            delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.54799, max=0.54799)
            delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.56021, max=0.56021)
            delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.55772, max=0.55772)

            adv_image = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(
                adv_image[0], adv_image[1], adv_image[2],
                json_absolute_path, weights_path, 0, model_current
            )
            if iterative_image_top1_label != labels:
                return adv_image
            return adv_image
