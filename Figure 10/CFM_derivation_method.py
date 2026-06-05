import torch
from torchvision import transforms
from PIL import Image
import os
from CFM import CFM
from CFM_Gaussain import CFM
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import GradCAM, show_cam_on_image
from feature_classifier.constants import *

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
def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

# Input the image into the model for category prediction, and input it as the path of the image file
def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with open(index_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = model_cnn(num_classes=num_classes).to(device)

    # Load model weights
    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()

    return(predicted_class_index, output[index])

# Calculate the pixel weight matrix of a single specified image with category index number "index"
def pixel_weight_matrix_image_path(image_path, weight_path, index, model_cnn):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = model_cnn(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()

    return channel_r, channel_g, channel_b

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eps_L0 = 0.2

# The maximum degree of tampering of a single pixel
degree = 255

# Iterative step size
iteration_step_size = 0.001

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

model_current = model_type

# The absolute path to the folder where the original images are stored


# Save the folder path for the generated attack samples


# The absolute path of the weight file


for i in range(50):
    file_list = os.listdir(folder_path+rf"\{i}")
    file_list = sorted(file_list, key=sort_func)


    success_num = 0
    image_num = 0
    # l2
    eps = float(np.sqrt((8 ** 2) * 3 * 224 * 224))
    os.makedirs(save_path_for_adversarial_samples+rf"\{i}", exist_ok=True)
    for file_name in file_list:
        image_num = image_num + 1
        actual_image_absolute_path = os.path.join(folder_path+rf"\{i}", file_name)
        image = Image.open(actual_image_absolute_path).resize((224, 224)).convert('RGB')




        actual_image = np.array(image)


        ######################################################


        ##########################################################

        # The R, G, B three channel matrix of the actual image
        actual_image_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        actual_image_matrix[0] = actual_image[:, :, 0]
        actual_image_matrix[0] = actual_image_matrix[0].astype(np.float64)
        actual_image_matrix[1] = actual_image[:, :, 1]
        actual_image_matrix[1] = actual_image_matrix[1].astype(np.float64)
        actual_image_matrix[2] = actual_image[:, :, 2]
        actual_image_matrix[2] = actual_image_matrix[2].astype(np.float64)

        actual_image_transform_matrix = np.zeros((3, 224, 224), dtype=np.float64)
        actual_image_transform_matrix[0] = ((actual_image_matrix[0] / 255) - 0.485) / 0.229
        actual_image_transform_matrix[1] = ((actual_image_matrix[1] / 255) - 0.456) / 0.224
        actual_image_transform_matrix[2] = ((actual_image_matrix[2] / 255) - 0.406) / 0.225

        image = data_transform(image).unsqueeze(0).cuda()

        # Load image
        img_absolute_path = actual_image_absolute_path
        assert os.path.exists(img_absolute_path), "file: '{}' dose not exist.".format(img_absolute_path)
        img = Image.open(img_absolute_path).convert('RGB')
        plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        # Read class_indict

        assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

        with open(json_absolute_path, "r") as f:
            class_indict = json.load(f)

        # Create model
        model = model_current(num_classes=50).to(device)

        # Load model weights
        weights_absolute_path = weights_path
        assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
        model.load_state_dict(torch.load(weights_absolute_path))

        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            top_probs, top_indices = torch.topk(predict, 5)

        # When generating adversarial samples for aimless attacks, it is necessary to set the label in "[]" to the true label of the image
        label = torch.tensor([top_indices[0]]).cuda()
        label_top2 = torch.tensor([top_indices[1]]).cuda()
        actual_image_index, x = predict_image_path(actual_image_absolute_path, json_absolute_path, weights_path, 0, model_current)

        atk = CFM(model, alpha=0.017, steps=max_num_iterative)
        # l2
        # atk = CFM(model,eps=eps, alpha=0.017, steps=max_num_iterative, mask=mask)
        adv_image = atk(image, label,label_top2, actual_image_transform_matrix, actual_image_matrix, actual_image)

        image_rgb = np.stack([adv_image[0], adv_image[1], adv_image[2]], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("No target attack image.png")
        iterative_image_path = "No target attack image.png"
        new_image_name = str(file_name)
        new_image_path = os.path.join(save_path_for_adversarial_samples,rf"{i}", new_image_name)
        image_pil.save(new_image_path)
