import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

sub = pd.read_csv('submission.csv')
def find_chars(img, img_path:str):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
    
    id = int(img_path.split('/')[-1].split('.')[0])
    flag = sub.loc[id, 'type']
    num = 0.5 if flag == 1 else 1
    # print(id, sub.loc[id, 'type'], num)
    H, W = gray.shape
    min_area = int(W*H*0.001*num)  
    boxes = []
    components = []
    stats = list(stats)
    stats.sort(key=lambda comp: comp[0])
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if i > 1 and area < min_area:
            if w < W*num*0.2*(1/(2*num)) and h < H*num*0.2:
                # print("width:",W, W*num*0.05*(1/num), w )
                # print("hieght:",H, H*num*0.2, h )
                # img2 = img.copy()
                # cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imshow('1', img2)
                # cv2.waitKey(0)
                # # print(i, stats[i], stats[i-1], stats[i-2], end=' \n ------------- \n')
                if (x > stats[i-1][0] and x < stats[i-1][0] + stats[i-1][2]
                    ) or (x > stats[i-2][0] and x < stats[i-2][0] + stats[i-2][2]):
                    continue
        components.append((x, y, w, h, area))

    components.sort(key=lambda comp: comp[0])
    for comp in components:
        x, y, w, h, area = comp
        if w > h:
            if w > h*3:
                # print(comp, min_area)
                x -= 15
                y -= int(15*2)
                w += 30
                h += int(30*2)
            else:
                x -= 7
                y -= 15
                w += 15
                h += 30
        elif h > w:
            x -= 10
            y -= 5
            w += 20
            h += 10
        else:
            x -= 10
            y -= 10
            w += 20
            h += 20
        boxes.append([x, y, w, h])

    return boxes


def predict_image(image, model, transform, device):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.mean(image, axis=2).astype(np.uint8) 
        image = Image.fromarray(image) 
    image = transform(image)  
    image = image.unsqueeze(0)  
    image = image.to(device)  

    model.eval()
    with torch.no_grad():  
        output = model(image) 
        _, predicted = torch.max(output, 1) 
    return predicted.item()  

def calculate(image_path, model):
    image  = cv2.imread(image_path)
    boxes = find_chars(image, image_path)
    text = ''
    for i in range(len(boxes)):
        x,y, w,h =boxes[i]
        img = image[y:y+h, x:x+w]

        predicted_class = predict_image(img, model, data_transform, device)

        label_map = {idx: label for idx, label in enumerate(sorted(pd.read_csv('ocr_data.csv')['label'].unique()))}
        predicted_label = label_map[predicted_class]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('1', image)
        cv2.waitKey(0)
        if predicted_label == 'divide':
            text += '/'
        elif predicted_label == 'x':
            text += '*'
        else:
            text += predicted_label
        # print(text)
    try:
        # print(round(eval(text),2), end='\n ------------------------- \n')
        output = round(eval(text),2)
    except:
        # print(0, end='\n ------------------------- \n')
        output = 0
    cv2.imshow('1', image)
    print(f"final output: {output}, {text}") 
    cv2.waitKey(0)
    formatted_num = "{:.2f}".format(output)
    return formatted_num 

class OCR_CNN(nn.Module):
    def __init__(self, num_classes=16):
        super(OCR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Fixed
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Fixed
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

def resize_with_padding(image, target_size=(64, 64), fill=255):
    w, h = image.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = image.resize((new_w, new_h), Image.LANCZOS)
    new_image = Image.new("L", target_size, fill)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def invert_if_black_bg(image):
    mean_value = np.array(image).mean() / 255.0
    if mean_value < 0.5:
        image = F.invert(image)
    return image

data_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_with_padding(img, (64, 64))),
    transforms.Lambda(lambda img: invert_if_black_bg(img)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_image(image, model, transform, device):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.mean(image, axis=2).astype(np.uint8) 
        image = Image.fromarray(image) 
    image = transform(image)  
    image = image.unsqueeze(0)  
    image = image.to(device)  

    model.eval()
    with torch.no_grad():  
        output = model(image) 
        _, predicted = torch.max(output, 1) 
    return predicted.item()  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
model = OCR_CNN(num_classes=16).to(device)
model.load_state_dict(torch.load('models/2-ocr_model.pth'))  
model.eval()  

for i in range(10, 40, 4):
    calculate(f'test/{i}.png',model)