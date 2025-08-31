import os
import shutil
import random
from itertools import islice
from collections import Counter

# Absolute paths
outputFolderPath = os.path.join("Dataset", "SplitData")
inputFolderPath = os.path.join("Dataset", "all")

splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# Remove and recreate output folder
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath)

# Create subdirectories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(outputFolderPath, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(outputFolderPath, split, "labels"), exist_ok=True)

# Get all valid base names
allFiles = os.listdir(inputFolderPath)
baseNames = list(set(name.split('.')[0] for name in allFiles))
random.shuffle(baseNames)

# Filter only files that have both .jpg and .txt
valid_baseNames = []
for name in baseNames:
    img_path = os.path.join(inputFolderPath, f"{name}.jpg")
    txt_path = os.path.join(inputFolderPath, f"{name}.txt")
    if os.path.exists(img_path) and os.path.exists(txt_path):
        valid_baseNames.append(name)

# Split data counts
lenData = len(valid_baseNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = lenData - lenTrain - lenVal

splits = [lenTrain, lenVal, lenTest]
Input = iter(valid_baseNames)
SplitData = [list(islice(Input, elem)) for elem in splits]
sequence = ['train', 'val', 'test']

# Copy files
for i, splitList in enumerate(SplitData):
    for name in splitList:
        shutil.copy(os.path.join(inputFolderPath, f"{name}.jpg"),
                    os.path.join(outputFolderPath, sequence[i], "images", f"{name}.jpg"))
        shutil.copy(os.path.join(inputFolderPath, f"{name}.txt"),
                    os.path.join(outputFolderPath, sequence[i], "labels", f"{name}.txt"))

# Create data.yaml
dataYaml = f"""path: {outputFolderPath}
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""

with open(os.path.join(outputFolderPath, "data.yaml"), 'w') as f:
    f.write(dataYaml)

print("Data split complete and data.yaml created.")
