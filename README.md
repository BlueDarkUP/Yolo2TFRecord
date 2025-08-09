# YOLO 到 TFRecord 转换器 (YOLO to TFRecord Converter)

这是一个功能强大且易于使用的 Python 脚本，用于将 [YOLO (You Only Look Once)](https://github.com/ultralytics/yolov5) 格式的图像标注数据集转换为 **TFRecord** 格式。转换后的数据集将严格遵循 [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 所要求的标准目录结构，并自动生成配套的 `label_map.pbtxt` 文件，为你省去繁琐的数据预处理步骤。

## 主要功能

- **解析 `data.yaml`**：自动从 YOLO 项目的 `data.yaml` 文件中读取训练集、验证集、测试集路径以及类别名称。
- **格式转换**：将 YOLO 格式的边界框标注（`class_id x_center y_center width height`）无缝转换为 TFRecord 格式所需的特征。
- **自动生成标签映射**：为每个数据集（train/valid/test）自动创建 `i_label_map.pbtxt` 文件，该文件是 TensorFlow 训练流程的必需品。
- **标准目录结构**：一键生成符合 TensorFlow Object Detection API 训练要求的目录结构，如下所示：
  ```
  <output_folder>/
  ├── train/
  │   ├── i.tfrecord
  │   └── i_label_map.pbtxt
  ├── valid/
  │   ├── i.tfrecord
  │   └── i_label_map.pbtxt
  └── test/
      ├── i.tfrecord
      └── i_label_map.pbtxt
  ```
- **鲁棒性设计**：能够优雅地处理测试集中不包含标签文件的情况，并对损坏的图像文件进行跳过处理。
- **进度可视化**：使用 `tqdm` 库显示转换进度条，方便追踪大数据集的处理过程。

## 环境要求

运行此脚本需要安装以下 Python 库：

- `tensorflow`
- `pyyaml`
- `pillow`
- `tqdm`

## 安装

你可以通过 pip 轻松安装所有必需的依赖项：

```bash
pip install tensorflow pyyaml pillow tqdm
```

## 如何使用

#### 步骤 1: 准备您的 YOLO 数据集

请确保你的 YOLO 数据集遵循标准的目录结构，并且包含一个 `data.yaml` 配置文件。

**数据集目录结构示例:**

```
/path/to/your_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

**`data.yaml` 文件内容示例:**

```yaml
train: ../datasets/coco128/images/train2017/
val: ../datasets/coco128/images/train2017/
test: 

# Classes
nc: 80  # number of classes
names: [ 'person', 'bicycle', 'car', 'motorcycle', ... ] # class names
```

#### 步骤 2: 配置转换脚本

打开此 Python 脚本 (`convert_yolo_to_tfrecord.py`)，找到文件末尾的 `if __name__ == '__main__':` 部分。根据你的实际情况，修改以下两个变量：

```python
if __name__ == '__main__':
    # --- 配置 ---
    # 1. 设置你的 data.yaml 文件路径
    # 例如: 'C:/datasets/coco128/data.yaml'
    yaml_file_path = './data.yaml' 
    
    # 2. 设置你希望保存最终 TFRecord 数据集的输出目录
    # 例如: 'C:/Users/YourUser/Downloads/tf_records'
    output_dataset_dir = './tfrecord_dataset'

    convert_yolo_to_tfrecord(yaml_file_path, output_dataset_dir)
```

#### 步骤 3: 运行转换脚本

在你的终端或命令行中，执行此 Python 脚本。

```bash
python convert_yolo_to_tfrecord.py
```

脚本运行后，你将会在指定的 `output_dataset_dir` 目录下看到一个结构清晰、可直接用于 TensorFlow 模型训练的数据集。

## 许可证

此项目采用 [MIT 许可证](LICENSE)。