import skimage.io as io
import os
from glob import glob
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


root = '/home/yeyang/hartenbach/runs/RESNET_DILATED/watch/11_23_1103_loss=L1_f=16_s=2_d=3_ck=3_norm=ln_lr=5e-4/SELECTED_PRED_4_INPAINTING'

image_files = glob(os.path.join(root, '*.png'))

tfrecords_filename = '../yy_data/firstTestResDilatedFHrunsSELECTEDmini.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for img_path in image_files:
    img = io.imread(img_path)

    height = img.shape[0]
    width = img.shape[1]
    # print(height)


    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw)}))

    # print(example['width'])

    writer.write(example.SerializeToString())

writer.close()
