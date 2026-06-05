parts=9
dataset_name="ImageNet-50"
folder_path = "/path/to/dataset/train"  # Path to training dataset folder
val_folder="/path/to/dataset/val"  # Path to validation dataset folder
save_path_for_adversarial_samples = "/path/to/save/adversarial_samples"  # Path to save generated adversarial samples
weights_path = "/path/to/model/weights.pth"  # Path to trained model weights
json_absolute_path = "/path/to/class_indices.json"  # Path to class indices JSON file
segment_folder="/path/to/segmented/images"  # Path to segmented images folder

# Model configuration
from torchvision.models import densenet121, resnet18, alexnet, vit_b_16, vit_b_32, googlenet, maxvit_t, vgg16
model_type = densenet121  # Model type (choose from imported models)
num_classes = 50  # Number of classification classes
model_save_path = "/path/to/save/model.pth"  # Path to save trained model
feature_model_save_path = "/path/to/save/cfm_enhanced_model.pth"  # Path to save feature extraction model
