import os
import io
import yaml
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


# --- 用于创建 TFRecord 特征的辅助函数 ---

def image_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_label_map_pbtxt(class_map, output_path):
    pbtxt_content = ""
    for class_id, class_name in sorted(class_map.items()):
        pbtxt_content += f"item {{\n"
        pbtxt_content += f"  id: {class_id + 1}\n"  # PBTXT 的 ID 通常从 1 开始
        pbtxt_content += f"  name: '{class_name}'\n"
        pbtxt_content += f"}}\n\n"

    with open(output_path, 'w') as f:
        f.write(pbtxt_content)
    print(f"成功创建标签映射文件: {output_path}")


def create_tf_example(image_path, label_path, class_map):
    if not os.path.exists(image_path):
        print(f"警告: 跳过，文件未找到。图像: {image_path}")
        return None

    labels_exist = os.path.exists(label_path)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    try:
        image = Image.open(io.BytesIO(encoded_jpg))
        if image.format != 'JPEG':
            with io.BytesIO() as output:
                image.convert('RGB').save(output, format="JPEG")
                encoded_jpg = output.getvalue()
    except (IOError, Image.DecompressionBombError):
        print(f"警告: 跳过损坏或有问题的图像: {image_path}")
        return None

    width, height = image.size

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    if labels_exist:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                xmin = x_center - box_width / 2.0
                xmax = x_center + box_width / 2.0
                ymin = y_center - box_height / 2.0
                ymax = y_center + box_height / 2.0

                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)

                # 注意：TFRecord 的类别 ID 通常从 1 开始，而 YOLO 从 0 开始。
                # 在这里我们保持 YOLO 的 0-based索引，在生成 pbtxt 时再加 1。
                class_name = class_map.get(class_id, "unknown")
                classes_text.append(class_name)
                classes.append(class_id + 1)  # 转换为 1-based 索引以匹配 pbtxt

    filename = os.path.basename(image_path)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_list_feature([height]),
        'image/width': int64_list_feature([width]),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': image_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))

    return tf_example


def convert_yolo_to_tfrecord(data_yaml_path, output_dir='.'):
    if not os.path.exists(data_yaml_path):
        print(f"错误: data.yaml 在 {data_yaml_path} 未找到")
        return

    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)

    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    class_names = data_config.get('names', [])
    class_map = {i: name for i, name in enumerate(class_names)}
    sets_to_process = {
        'train': data_config.get('train'),
        'valid': data_config.get('val'),
        'test': data_config.get('test')
    }

    for set_name, image_dir_rel in sets_to_process.items():
        if not image_dir_rel:
            print(f"跳过 '{set_name}' 集: 在 data.yaml 中未定义路径")
            continue

        # 为每个数据集创建子目录
        set_output_dir = os.path.join(output_dir, set_name)
        os.makedirs(set_output_dir, exist_ok=True)

        # 生成 label_map.pbtxt
        label_map_path = os.path.join(set_output_dir, 'i_label_map.pbtxt')
        create_label_map_pbtxt(class_map, label_map_path)

        image_dir = os.path.join(base_dir, image_dir_rel)
        label_dir = image_dir.replace('images', 'labels')

        if not os.path.exists(image_dir):
            print(f"错误: '{set_name}' 的图像目录未找到: {image_dir}")
            continue

        if set_name != 'test' and not os.path.exists(label_dir):
            print(f"错误: '{set_name}' 的标签目录未找到: {label_dir}")
            continue
        elif set_name == 'test' and not os.path.exists(label_dir):
            print(f"信息: 'test' 集的标签目录未找到: {label_dir}。将在没有标签的情况下继续。")

        output_filename = os.path.join(set_output_dir, 'i.tfrecord')
        print(f"开始转换 '{set_name}' 数据集...")
        print(f"输出将保存至: {output_filename}")

        with tf.io.TFRecordWriter(output_filename) as writer:
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for image_file in tqdm(image_files, desc=f"正在处理 {set_name} 图像"):
                image_path = os.path.join(image_dir, image_file)
                label_filename = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(label_dir, label_filename)

                tf_example = create_tf_example(image_path, label_path, class_map)

                if tf_example:
                    writer.write(tf_example.SerializeToString())

        print(f"成功创建 {output_filename}")
        print("-" * 30)


if __name__ == '__main__':
    yaml_file_path = './data.yaml'
    output_dataset_dir = './tfrecord_dataset'

    convert_yolo_to_tfrecord(yaml_file_path, output_dataset_dir)