import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from PIL import Image
import json
import torchvision.transforms as transforms
import numpy as np
import z3
import sys
import glob
from collections import OrderedDict


class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self._attacks = OrderedDict()
        self.set_model(model)
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None
        self.normalization_used = None
        self._normalization_applied = None
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError

    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        logits = self.model(inputs)
        return logits

    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    def set_device(self, device):
        self.device = device

    def set_normalization_used(self, mean, std):
        self.normalization_used = {}
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._set_normalization_applied(True)

    def normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def __call__(self, inputs, labels=None, *args, **kwargs):
        given_training = self.model.training
        if not self._model_training:
            self.model.eval()
        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
        if given_training:
            self.model.train()
        return adv_inputs


def map_to_224x224(input_matrix, n, output_channels=3):
    assert input_matrix.shape == (n, n), f"Input matrix shape must be({n}, {n})"
    if output_channels == 1:
        output_matrix = np.zeros((224, 224), dtype=np.float32)
    else:
        output_matrix = np.zeros((output_channels, 224, 224), dtype=np.float32)

    h_splits = np.array_split(range(224), n)
    w_splits = np.array_split(range(224), n)

    for i in range(n):
        for j in range(n):
            h_start, h_end = h_splits[i][0], h_splits[i][-1] + 1
            w_start, w_end = w_splits[j][0], w_splits[j][-1] + 1
            if output_channels == 1:
                output_matrix[h_start:h_end, w_start:w_end] = input_matrix[i, j]
            else:
                for c in range(output_channels):
                    output_matrix[c, h_start:h_end, w_start:w_end] = input_matrix[i, j]
    return output_matrix


def split_into_blocks(matrix, n):
    assert matrix.shape == (3, 224, 224), "Input matrix shape must be(3, 224, 224)"
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


def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):
    if isinstance(actual_image_matrix, torch.Tensor):
        actual_image_matrix = actual_image_matrix.detach().cpu().numpy()
    adv_image = actual_image_matrix.copy()

    if isinstance(adversarial_sample_transform_matrix, torch.Tensor):
        adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.detach().cpu().numpy()

    if isinstance(actual_image_transform_matrix, torch.Tensor):
        actual_image_transform_matrix = actual_image_transform_matrix.detach().cpu().numpy()

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


channel_clamps = {
    0: {"min": -2.1179, "max": 2.2489},
    1: {"min": -2.0357, "max": 2.4285},
    2: {"min": -1.8044, "max": 2.64}
}


class CFM(Attack):
    def __init__(self, model, eps=float(np.sqrt((8 ** 2) * 3 * 224 * 224)), alpha=1 / 255, steps=10, random_start=True,
                 vcr=0.9, mask=np.zeros((3, 224, 224), dtype=np.float32), parts=8):
        super().__init__("CFM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.vcr = vcr
        self.mask = mask
        self.parts = parts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def apply_brighten(self, matrix, brightness_addition=0.1):
        result = matrix.copy()
        for k in range(3):
            result[k] = np.clip(result[k] + brightness_addition, 
                                channel_clamps[k]["min"], 
                                channel_clamps[k]["max"])
        return result

    def forward(self, images, labels, labels_top2, actual_image_transform_matrix, actual_image_matrix, actual_image):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels_top2 = labels_top2.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        mask = torch.from_numpy(self.mask).to(self.device)

        specific_values = [-0.1, 0, 0.1]
        
        if isinstance(actual_image_transform_matrix, torch.Tensor):
            adv_image_transform_matrix = actual_image_transform_matrix.detach().cpu().numpy().copy()
        else:
            adv_image_transform_matrix = actual_image_transform_matrix.copy()

        for step in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, retain_graph=True, create_graph=False)[0]

            cost_top2 = loss(outputs, labels_top2)
            grad_top2 = torch.autograd.grad(cost_top2, adv_images, retain_graph=False, create_graph=False)[0]


            grad_np = grad.squeeze(0).cpu().numpy()
            grad_top2_np = grad_top2.squeeze(0).cpu().numpy()

            splited_grad = split_into_blocks(grad_np, self.parts)
            splited_grad_max = np.zeros((self.parts, self.parts))
            splited_top2_grad = split_into_blocks(grad_top2_np, self.parts)
            splited_top2_grad_max = np.zeros((self.parts, self.parts))

            for i in range(self.parts):
                for j in range(self.parts):
                    splited_grad_max[i][j] = np.average(splited_grad[i][j])
                    splited_top2_grad_max[i][j] = np.average(splited_top2_grad[i][j])

            solver = z3.Optimize()
            variables = [[0 for j in range(self.parts)] for i in range(self.parts)]
            abs_sum = z3.Sum([z3.Abs(variables[i][j]) for i in range(self.parts) for j in range(self.parts)])

            for i in range(self.parts):
                for j in range(self.parts):
                    var = z3.Real(f'x_{i}_{j}')
                    variables[i][j] = var
                    value_constraints = [var == val for val in specific_values]
                    solver.add(z3.Or(value_constraints))
                    if splited_grad_max[i][j] != splited_top2_grad_max[i][j]:
                        solver.add(var * splited_grad_max[i][j] > var * splited_top2_grad_max[i][j])

            solver.minimize(abs_sum)
            check_result = solver.check()

            if check_result == z3.sat:
                model_z3 = solver.model()
                result_matrix = np.zeros(shape=(self.parts, self.parts))

                for i in range(self.parts):
                    for j in range(self.parts):
                        var = variables[i][j]
                        if isinstance(var, z3.ExprRef):
                            var_value = float(model_z3[var].as_decimal(prec=10).rstrip('?'))
                        else:
                            var_value = 0.0
                        result_matrix[i][j] = var_value

                h_splits = np.array_split(range(224), self.parts)
                w_splits = np.array_split(range(224), self.parts)

                brightness_addition = 255
                adv_image_transform_matrix_brightened = self.apply_brighten(adv_image_transform_matrix.copy(), brightness_addition)

                for blk_i in range(self.parts):
                    for blk_j in range(self.parts):
                        if result_matrix[blk_i][blk_j] <= 0:
                            h_start, h_end = h_splits[blk_i][0], h_splits[blk_i][-1] + 1
                            w_start, w_end = w_splits[blk_j][0], w_splits[blk_j][-1] + 1
                            for k in range(3):
                                adv_image_transform_matrix_brightened[k, h_start:h_end, w_start:w_end] = actual_image_transform_matrix[k, h_start:h_end, w_start:w_end]

                adv_image_transform_matrix = adv_image_transform_matrix_brightened

                m = (adv_image_transform_matrix == actual_image_transform_matrix)

                for ch in range(3):
                    ch_min = channel_clamps[ch]["min"]
                    ch_max = channel_clamps[ch]["max"]
                    adv_image_transform_matrix[ch] = np.clip(adv_image_transform_matrix[ch], ch_min, ch_max)

                adv_images = torch.tensor(adv_image_transform_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
                adv_images.requires_grad = True

                adv_image = adv_images.squeeze(0).cpu()
                adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)

                return adv_image

            elif check_result == z3.unsat:
                break
            else:
                break

        adv_image = adv_images.squeeze(0).cpu()
        adv_image = con_transform(actual_image_transform_matrix, adv_image, actual_image_matrix)
        return adv_image


def main():
    input_dir = PATH_PLACEHOLDER
    output_dir = PATH_PLACEHOLDER
    
    os.makedirs(output_dir, exist_ok=True)

    import torchvision.models as models
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    weights_path = PATH_PLACEHOLDER
    
    model = models.densenet121(weights=None)
    state_dict = torch.load(weights_path, map_location=device)
    
    model.classifier = nn.Linear(model.classifier.in_features, 50)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_patterns = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not image_files:
        return


    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        image = Image.open(img_path).resize((224, 224)).convert('RGB')
        actual_image = np.array(image)

        actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        actual_image_matrix[0] = actual_image[:, :, 0]
        actual_image_matrix[1] = actual_image[:, :, 1]
        actual_image_matrix[2] = actual_image[:, :, 2]

        actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
        actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
        actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            predict = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(predict, 2)

        label = top_indices[0][0]
        label_top2 = top_indices[0][1]

        atk = CFM(model, alpha=0.017, steps=10, parts=9)
        adv_image = atk(image_tensor, label.unsqueeze(0), label_top2.unsqueeze(0), 
                        actual_image_transform_matrix, actual_image_matrix, actual_image)

        image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)

        output_path = os.path.join(output_dir, f"{base_name}_CFM_brightened.png")
        image_pil.save(output_path)


if __name__ == '__main__':
    main()