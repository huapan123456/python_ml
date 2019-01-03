import tensorflow as tf
import os


# 模拟一下同步先处理数据，然后才能取数据训练
# tensorflow当中，运行操作有依赖性

# # 1、首先定义队列
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 放入一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 2、定义一些处理数据的螺距，取数据的过程      取数据，+1， 入队列
#
# out_q = Q.dequeue()
#
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#     # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# 模拟异步子线程 存入样本， 主线程 读取样本

# # 1、定义一个队列，1000
# Q = tf.FIFOQueue(1000, tf.float32)
#
# # 2、定义要做的事情 循环 值，+1， 放入队列当中
# var = tf.Variable(0.0)
#
# # 实现一个自增  tf.assign_add
# data = tf.assign_add(var, tf.constant(1.0))
#
# en_q = Q.enqueue(data)
#
# # 3、定义队列管理器op, 指定多少个子线程，子线程该干什么事情
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 初始化变量的OP
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     # 开启线程管理器
#     coord = tf.train.Coordinator()
#
#     # 真正开启子线程
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程，不断读取数据训练
#     for i in range(300):
#         print(sess.run(Q.dequeue()))
#
#     # 回收你
#     coord.request_stop()
#
#     coord.join(threads)


# 批处理大小，跟队列，数据的数量没有影响，只决定 这批次取多少数据


def csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1、构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造csv阅读器读取队列数据（按一行）
    reader = tf.TextLineReader()

    key, value = reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [["None"], ["None"]]

    example, label = tf.decode_csv(value, record_defaults=records)

    # 4、想要读取多个数据，就需要批处理
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    print(example_batch, label_batch)
    return example_batch, label_batch


def picread(filelist):
    """
    读取狗图片并转换成张量
    :param filelist: 文件路径+ 名字的列表
    :return: 每张图片的张量
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    print(value)

    # 3、对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)

    print(image)

    # 5、处理图片的大小（统一大小）
    image_resize = tf.image.resize_images(image, [200, 200])

    print(image_resize)

    # 注意：一定要把样本的形状固定 [200, 200, 3],在批处理的时候要求所有数据形状必须定义
    image_resize.set_shape([200, 200, 3])

    print(image_resize)

    # 6、进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)

    print(image_batch)

    return image_batch


# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./tmp/cifar.tfrecords", "存进tfrecords的文件")


class CifarRead(object):
    """完成读取二进制文件， 写进tfrecords，读取tfrecords
    """
    def __init__(self, filelist):
        # 文件列表
        self.file_list = filelist

        # 定义读取的图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制文件每张图片的字节
        self.label_bytes = 1
        self.image_bytes = self.height * self.width * self.channel
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):

        # 1、构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2、构造二进制文件读取器，读取内容, 每个样本的字节数
        reader = tf.FixedLengthRecordReader(self.bytes)

        key, value = reader.read(file_queue)

        # 3、解码内容, 二进制文件内容的解码
        label_image = tf.decode_raw(value, tf.uint8)

        print(label_image)

        # 4、分割出图片和标签数据，切除特征值和目标值
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)

        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        # 5、可以对图片的特征数据进行形状的改变 [3072] --> [32, 32, 3]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        print(label, image_reshape)
        # 6、批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        print(image_batch, label_batch)
        return image_batch, label_batch

    def write_ro_tfrecords(self, image_batch, label_batch):
        """
        将图片的特征值和目标值存进tfrecords
        :param image_batch: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        # 1、建立TFRecord存储器
        writer = tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)

        # 2、循环将所有样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            # 取出第i个图片数据的特征值和目标值
            image = image_batch[i].eval().tostring()

            label = int(label_batch[i].eval()[0])

            # 构造一个样本的example
            example =  tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            # 写入单独的样本
            writer.write(example.SerializeToString())

        # 关闭
        writer.close()
        return None

    def read_from_tfrecords(self):

        # 1、构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        # 2、构造文件阅读器，读取内容example,value=一个样本的序列化example
        reader = tf.TFRecordReader()

        key, value = reader.read(file_queue)

        # 3、解析example
        features = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })

        # 4、解码内容, 如果读取的内容格式是string需要解码， 如果是int64,float32不需要解码
        image = tf.decode_raw(features["image"], tf.uint8)

        # 固定图片的形状，方便与批处理
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        label = tf.cast(features["label"], tf.int32)

        print(image_reshape, label)

        # 进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch


if __name__ == "__main__":
    # 1、找到文件，放入列表   路径+名字  ->列表当中
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    # print(file_name)
    cf = CifarRead(filelist)

    # image_batch, label_batch = cf.read_and_decode()

    image_batch, label_batch = cf.read_from_tfrecords()

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 存进tfrecords文件
        # print("开始存储")
        #
        # cf.write_ro_tfrecords(image_batch, label_batch)
        #
        # print("结束存储")

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()

        coord.join(threads)


