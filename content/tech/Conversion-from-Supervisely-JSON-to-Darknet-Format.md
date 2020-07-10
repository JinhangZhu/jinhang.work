---
title: "Conversion From Supervisely JSON to Darknet Format"
date: 2020-05-07T22:15:58+01:00
categories: [Tech,Object detection]
tags: [JSON, Darknet]
slug: "conversion-from-supervisely-json-to-darknet-format"
---

How to convert the JSON files exported by Supervisely annotation tool to the Darknet format in python? *GitHub repo*: [JinhangZhu/supervisely-to-darknet](https://github.com/JinhangZhu/supervisely-to-darknet)<!--more-->

## JSON

JSON (JavaScript Object Notation) is a popular data format for storing structured data. I don't care what is could be done and other stuff about it. What I need to do is to make this format converted to the format that my YOLOv3 model can take. But first thing first, let's see what format JSON has.

### Syntax rules

- Data is in name/value pairs.
- Data is separated by commas `,`.
- Curly braces `{}` hold objects.
- Square brackets `[]` hold arrays.

### Components

- Name/value pairs. A data pair consist of a **field** name in double quotes `""`, followed by a colon `:`, followed by a value: `"classTitle": "left_hand"`

- Objects. Objects are written inside curly braces `{}`. There may be multiple name/value pairs inside one pair of curly braces, just like what dictionaries look like in Python.

  ```json
  {"firstName":"John", "lastName":"Doe"}
  ```

- Arrays. JSON arrays are written in square brackets `[]`. Like the list in Python, an array can contain objects.

  ```json
  "employees":[ // The obejct "employees" is an array that contains three objects
      {"firstName":"John", "lastName":"Doe"},
      {"firstName":"Anna", "lastName":"Smith"},
      {"firstName":"Peter", "lastName":"Jones"}
  ]
  ```

## Python JSON

In Python, JSON exists as a string. For example:

```python
p = '{"name": "Bob", "languages": ["Python", "Java"]}'
```

To work with JSON (string, or file containing JSON object), we use Python's `json` module.

```python
import json
```

### Parse JSON to dict

To parse a JSON string, we use `json.loads()` method, which returns a dictionary. For example:

```python
import json

person = '{"name": "Bob", "languages": ["English", "Fench"]}'
person_dict = json.loads(person)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print( person_dict)

# Output: ['English', 'French']
print(person_dict['languages'])
```

### Read JSON file

Our Supervisely annotations are stored in JSON files, so we need to load the file first. For example, say a `.json` file contains a JSON object:

```json
{"name": "Bob", 
"languages": ["English", "Fench"]
}
```

And we parse the file:

```python
import json

with open('path_to_file/person.json') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)
```

## Supervisely format

Supervisely json-based annotation format supports several figures including: `rectangle`, `line`, `polygon`.... BUT we only care about the rectangular objects.

### JSON for the whole project

Each project has predefined objects classes and tags. File `meta.json` contains this information. Ours is as follows:

```json
{
    "classes": [
        {
            "title": "right_hand",
            "shape": "rectangle",
            "color": "#CE933B",
            "geometry_config": {}
        },
        {
            "title": "left_hand",
            "shape": "rectangle",
            "color": "#D92677",
            "geometry_config": {}
        }
    ],
    "tags": []
}
```

- "classes": list of objects - all possible object classes.
  - "title": string - the unique identifier of a class - the name of the class.
  - "shape": string - annotation shape.
  - "color": string - hex color code (not important here)
  - ...

### JSON for an image

For each image we keep a json file with annotations.

```json
{
    "description": "",
    "tags": [],
    "size": {
        "height": 1080,
        "width": 1920
    },
    "objects": [
        {
            "description": "",
            "geometryType": "rectangle",
            "labelerLogin": "sunnyluyao",
            "tags": [],
            "classTitle": "left_hand",
            "points": {
                "exterior": [
                    [
                        751,
                        684
                    ],
                    [
                        970,
                        1047
                    ]
                ],
                "interior": []
            }
        },
        {
            "description": "",
            "geometryType": "rectangle",
            "labelerLogin": "sunnyluyao",
            "tags": [],
            "classTitle": "right_hand",
            "points": {
                "exterior": [
                    [
                        1131,
                        796
                    ],
                    [
                        1365,
                        1080
                    ]
                ],
                "interior": []
            }
        }
    ]
}
```

- "size": is equal to image size.

  - "width": image width in pixels
  - "height": image height in pixels

- "objects": list of objects, contains fields about the annotated label rectangles with their values.

  - "classTitle": string - the name of the class. It is used to identify the class shape from the `meta.json`.

  - "points": object with two fields:

    - "exterior": list of two lists with two numbers (coordinates):

      `[[left, top], [right, bottom]]`

    - "interior": always empty for rectangles.

## Darknet format

Darknet format specifies not only the annotation format for each image but also the format of files in folders. We follow the format of COCO: images and labels are in separate parallel folders, and one label file per image (if no objects in image, o label file is required). 

### Label files

The label file specifications are:

- One row per object

- Each row is in the format of `class b_x_center b_y_center b_width b_height`.

- Box coordinates must be in **normalised** xywh format (from 0 to 1). Since Supervisely coordinates are in pixels, normalisation step is required on both x and y axes.

  Say $x_{LT}, y_{LT}, x_{RB}, y_{RB}$ are respectively the elements in `[[left, top], [right, bottom]]`. $height, width$ are image sizes. Then normalisation is like:
  $$
  \text{b_x_center} = \frac{x_{LT}+x_{RB}}{2\times width}\\\\
  \text{b_y_center} = \frac{y_{LT}+y_{RB}}{2\times height}\\\\
  \text{b_width} = \frac{x_{RB}-x_{LT}}{width}\\\\
  \text{b_height} = \frac{y_{RB}-y_{LT}}{height}
  $$

- Class numbers are zero-indexed (start from 0).

For example, one-row label:

```
1 0.5841911764705883 0.535625 0.030147058823529412 0.04375
```

Each image's label file must be locatable by simply replacing `/images/*.jpg` with `/labels/*.txt` in its path name.

### Data splitting

There should be a `.txt` file that contains the locations of images of the dataset. Each row contains a path to an image, and remember **one label must also exist in a corresponding `/labels` folder for each image containing objects.**

![](https://user-images.githubusercontent.com/26833433/78174735-95cfa900-740e-11ea-8e50-bfa7e934e768.png)

### `.names` file

The file lists all the class names in the dataset. Each row contains one class name.

### `.data` file

There should be class count (e.g. COCO has 80 and P30 has 2), paths to train and validation datasets (the `.txt` files mentioned above), and the path to the `.names` file. 

## Coding

Firstly, we create a sub folder called "./dataset/" (or something else), which will contains all our data generated.

```python
# Create folders: images and labels
# https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/utils.py#L73
def make_folders(path='./dataset/'):
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
```

Then, we need to know what classes the whole image set has. This information is easily found in `meta.json`, so we import the json file and read the values of the key "classes". After reading, the class names are appended into the `.names` file.

```python
# Get names of classes from meta.json and write them into *.names file
# In reference to https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/run.py#L36
def get_classes(read_name, write_name=None):
    # Import JSON
    with open(read_name) as f:
        data = json.load(f)

    # Get classes from "classes" - "title" values
    classes_object = data['classes']
    classes = []
    for class_object in tqdm(classes_object, desc="Names of classes"):
        class_name = class_object['title']
        classes.append(class_name)
        
        # Write *.names file
        with open(write_name + 'classes.names', 'a') as nf:
            nf.write('{}\n'.format(class_name))
    return classes
```

We create a function called `conver_supervisely_json()` that performs making folders and obtaining classes before writing labels.

```python
# Convert from Supervisely format to darknet format.
# In reference to https://github.com/ultralytics/JSON2YOLO/blob/177e96ad79bb1832c82dc3a1cec6681329ee1835/run.py#L10
def convert_supervisely_json(read_path, new_data_name, meta_file):
    # Create folders
    out_path = './' + new_data_name + os.sep
    make_folders(out_path)

    # Write classes.names from meta.json
    classes = get_classes(meta_file, out_path)
```

As Supervisely exports images and annotation files in separate folders `img` and `ann`, we use `glob` to obtain the iterable paths of files withing two folders and then sort them.

```python
    # Get all file real paths
    name = name + os.sep
    ann_paths = sorted(glob.glob(name + 'ann/' + '*.json'))
    img_paths = sorted(glob.glob(name + 'img/' + '*.jpg'))
```

It is now time to import the json files and read data from them. We will assign each image **with at least one object** a label file in `.txt` in the labels folder. For each object bounding box, there should be class index (in integer numbers), normalised center coordinates and normalised size. We also copy the images to the images folder.

```python
    # Import all json annotation files for images
    for (ann_path, img_path) in tqdm(zip(ann_paths, img_paths), desc='Annotations'):
        label_name = os.path.basename(img_path) + '.txt'

        # Import json
        with open(ann_path) as ann_f:
            ann_data = json.load(ann_f)
        
        # Image size
        image_size = ann_data['size']   # dict: {'height': , 'width': }

        # Objects bounding boxes
        bboxes = ann_data['objects']
        for bbox in tqdm(bboxes, desc='Bounding boxes'):
            class_index = classes.index(bbox['classTitle'])
            corner_coords = bbox['points']['exterior']  # bbox corner coordinates in [[left, top], [right, bottom]]

            # Normalisation
            b_x_center = (corner_coords[0][0] + corner_coords[1][0]) / 2 / image_size['width']
            b_y_center = (corner_coords[0][1] + corner_coords[1][1]) / 2 / image_size['height']
            b_width = (corner_coords[1][0] - corner_coords[0][0]) / image_size['width']
            b_height = (corner_coords[1][1] - corner_coords[0][1]) / image_size['height']

            # Write labels file
            if (b_width > 0.) and (b_height > 0.):
                with open(out_path + 'labels/' + label_name, 'a') as label_f:
                    label_f.write('%d %.6f %.6f %.6f %.6f\n' % (class_index, b_x_center, b_y_center, b_width, b_height))
        
        # Move images to images folder
        shutil.copy(img_path, out_path + 'images/')
```

Run the function and we see 754 label files in the labels folder, as indicated by the Supervisely filter that the number of images with at least one object is correct.

<img src="https://i.loli.net/2020/05/03/MANupymBVqZho2a.png" style="zoom:80%;" />

As YOLOv3 requires a train set and a validation set in the form of **collections of path names**. We need to create two `*.txt` files to contain separate sets of paths of images withing `./dataset/images` folder. This feature is supposed to be implemented by:

```python
    # Split training set
    img_paths = sorted(glob.glob(out_path + 'images/' + '*.jpg'))
    split_paths(new_data_name, img_paths)
```

Dataset splitting is transformed to **elements splitting**, i.e. we randomly choose different sizes of collections of elements from the whole set of pathnames. By randomly choosing indices and then we can use the indices to select the separate sets of paths.

```python
# Random split: get random indices
def split_indices(data, train_ratio=0.9, val_ratio=0.1, shuffle=True):
    test_ratio = 1 - train_ratio - val_ratio
    indices = np.arange(len(data))
    if shuffle == True:
        np.random.shuffle(indices)
    end_train = round(len(data) * train_ratio)
    end_val = round(len(data) * val_ratio + end_train)
    end_test = round(len(data) * test_ratio + end_val)

    return indices[:end_train], indices[end_train:end_val], indices[end_val:end_test]
```

Split the paths into several sets according to **whether the corresponding indices are empty or not**:

```python
# Random split: split the paths
def split_paths(new_data_name, img_paths):
    out_path = './' + new_data_name + os.sep
    train_ids, val_ids, test_ids = split_indices(img_paths, 0.9, 0.1, True)
    datasets = {'train': train_ids, 'validation': val_ids, 'test': test_ids}
    for key, ids in datasets.items():
        if ids.any():
            with open(out_path + new_data_name + '_' + key + '.txt', 'a') as wf:
                for idx in tqdm(ids, desc=key + ' paths'):
                    wf.write('{}'.format(img_paths[idx]) + '\n')
```

Run the code:

```python
convert_supervisely_json(
    read_path='P30__P30_04',
    new_data_name='P30',
    meta_file='meta.json'
)
```

 and we'll see files withing the folder we specified:

```bash
$ ls
classes.names  images  labels  P30_train.txt  P30_validation.txt
```

## References

- [What is JSON?](https://www.w3schools.com/whatis/whatis_json.asp)
- [Python JSON](https://www.programiz.com/python-programming/json)
- [Supervisely Format](https://docs.supervise.ly/data-organization/import-export/supervisely-format)
- [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data)