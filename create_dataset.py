import os
import sys
import numpy as np
import tensorflow as tf


def txt2tfrecords(data_path, save_path):
    raw_dataset = np.reshape(np.loadtxt(data_path), (-1, 6, 5))
    print(data_path, ': ', raw_dataset.shape, sep='')
    writer = tf.io.TFRecordWriter(os.path.join(save_path, os.path.basename(data_path).split('.')[0] + '.tfrecords'))
    for data in raw_dataset:
        feature = {}
        for d, n in zip([data], ['data']):
            feature[n] = tf.train.Feature(float_list=tf.train.FloatList(value=d.flatten()))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    print('done: save to ', os.path.join(save_path, os.path.basename(data_path).split('.')[0] + '.tfrecords'), sep='')


if '__main__' == __name__:
    txt2tfrecords(sys.argv[1], "./dataset/")
