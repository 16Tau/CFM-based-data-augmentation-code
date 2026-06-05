import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
# from test_tools.notification import Notification
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from constants import *
# from eval_model_behavior_function import eval_behavior


best_behavior=False

class MultiRootFolderDataset(Dataset):
    def __init__(self, root_folders, transform=None):
        """
        Custom dataset to load images from multiple root folders, where each root folder contains subfolders as classes.
        :param root_folders: List of root folder paths, each containing subfolders named as class labels.
        :param transform: Transformations to apply to the images.
        """
        self.root_folders = root_folders
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}  # Map class names to unique indices
        self.idx_to_class = {}  # Reverse mapping for debugging or later use

        # Collect image paths and labels
        unique_class_idx = 0
        for root_folder in root_folders:
            if not os.path.isdir(root_folder):
                raise ValueError(f"Root folder does not exist: {root_folder}")

            for class_name in os.listdir(root_folder):
                class_path = os.path.join(root_folder, class_name)
                if os.path.isdir(class_path):
                    # Assign a unique index to each class
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = unique_class_idx
                        self.idx_to_class[unique_class_idx] = class_name
                        unique_class_idx += 1

                    # Collect images from the current subfolder
                    for filename in os.listdir(class_path):
                        file_path = os.path.join(class_path, filename)
                        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
                            self.image_paths.append(file_path)
                            self.labels.append(self.class_to_idx[class_name])

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in provided root folders: {root_folders}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")  # Ensure 3-channel images

        if self.transform:
            img = self.transform(img)

        return img, label
# # Hyperparameters
# c_list=[5]  ## Number of tampering iterations
l_list =[0.1]  ## Tampering step size
if_partial=False
partial=0.25
def main():

    loss_sequence=[]
    attention_sequence=[]
    for l in l_list:
        # Set the device based on CUDA availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Perform initialization operations on the images in the training and validation sets
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            ,"val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        }

        # Import the training dataset


        # train_dataset = datasets.ImageFolder(root="../datasets/cifar10/train",
        #                                      transform=data_transform["train"])
        folder_list=["../datasets/Imagenet-50/train", "../datasets/cifar10/new_train/split_both/1"]
        folder_list=[folder_path,save_path_for_adversarial_samples]
        # folder_list=["../datasets/VOC2007/classification/train"]
        train_dataset = MultiRootFolderDataset(root_folders=folder_list, transform=data_transform["train"])
        train_num = len(train_dataset)

        # # Convert the class labels and index positions of the training dataset into a dictionary
        # classes_list = train_dataset.class_to_idx
        # cla_dict = dict((val, key) for key, val in classes_list.items())
        # # Write dict into json file
        # json_str = json.dumps(cla_dict, indent=4)
        # with open('class_indices.json', 'w') as json_file:
        #     json_file.write(json_str)

        batch_size = 32
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=nw)

        validate_dataset = datasets.ImageFolder(root=val_folder,
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=4, shuffle=False,
                                                      num_workers=nw)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))

        # Creates a model object using the AlexNet architecture for a binary classification task
        # net = VGG(num_classes=3, init_weights=True)
        net = model_type(num_classes=num_classes)
        # net = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        # net.classifier[6] = nn.Linear(in_features=4096, out_features=3)
        net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0002)
        # Training rounds
        epochs = 100


        # Weight file
        save_path = model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_acc = 0.0
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # Train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            epoch_loss = running_loss / len(train_loader)
            loss_sequence.append(epoch_loss)
            # Validate
            net.eval()
            acc = 0.0

            if not best_behavior:
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = net(val_images.to(device))
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_accurate = acc / val_num
                    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                          (epoch + 1, running_loss / train_steps, val_accurate))
            else:
                temp_path=os.path.join(os.path.dirname(save_path),"temp.pth")
                torch.save(net.state_dict(), temp_path)
                # val_accurate=eval_behavior(temp_path)
                print('[epoch %d] train_loss: %.3f  net_attention: %.3f' %
                      (epoch + 1, running_loss / train_steps, val_accurate))
                attention_sequence.append(val_accurate)
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
        plt.clf()
        plt.plot(range(len(loss_sequence)), loss_sequence, marker="o")
        if len(attention_sequence)>0:
            plt.plot(range(len(attention_sequence)), attention_sequence, marker="o",color='r')
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.show()
        del net
    # message = Notification(['Finished Training'])
    # message.send()
if __name__ == '__main__':
    main()


