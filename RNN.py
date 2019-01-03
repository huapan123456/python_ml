import collections
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("poetry_file", "./data/poetry/poetry.txt", "诗集目录")


class RobotPoetry(object):
    """自动写诗机器人
    """

    def __init__(self, batch_size=64):
        # 总的词数以及词到数字的映射
        self.VOCAB_SIZE = 0
        self.word_num_map = {}

        # 用于训练的输入数据和输出数据，n个batch_size大小组成一个列表
        self.x_batches = []
        self.y_batches = []

        # RNN训练的每批次样本数量
        self.batch_size = batch_size
        # 训练步数
        self.max_step = 0

        # 隐层的数量
        self.rnn_size = 100

        # 定义占位符，供运行时提供数据
        # 定义输入层,可以看到输入层的维度为batch_size*num_steps,这个num_steps为序列长度未知
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, None])
        # 定义正确输出结果
        self.output_targets = tf.placeholder(tf.int32, [self.batch_size, None])

    def get_poetry(self):

        poetrys = []

        # 一、解析文件当中的诗，进行处理，放到包含所有诗的列表当中

        # 处理文件每行格式
        with open(FLAGS.poetry_file, "r", encoding='utf-8', ) as f:
            for line in f:
                try:
                    # 分割诗名和诗体内容
                    title, content = line.strip().split(':')
                    # 去除空格
                    content = content.replace(' ', '')
                    # 如果诗中有一些非法字符则跳过（视情况而定）
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                        continue
                    # 确定诗的长度大小在一定范围之内
                    if len(content) < 5 or len(content) > 79:
                        continue
                    # 给一首诗加上头、尾标记
                    content = '[' + content + ']'
                    poetrys.append(content)
                except Exception as e:
                    pass
        # 按每句诗的字数(长度)排序
        poetrys = sorted(poetrys, key=lambda line: len(line))
        # 统计每个字出现次数
        all_words = []
        for poetry in poetrys:
            all_words += [word for word in poetry]

        # 输出一个字典： key是word， value是这个word出现的次数
        counter = collections.Counter(all_words)
        # 返回一个tuple列表， tuple是(key, value), 按x[1] 即value的降序
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # 将每个词提取出来放到元组当中
        words, _ = zip(*count_pairs)

        # 二、建立字的数字映射

        # 取出所有词作为总的使用的字的列表
        self.VOCAB_SIZE  = words[:len(words)] + (' ',)

        # 每个字映射为一个数字ID
        self.word_num_map = dict(zip(self.VOCAB_SIZE, range(len(self.VOCAB_SIZE))))

        # 三、将诗转换成向量的形式

        # 把一首诗转换成单个词的数字替代
        to_num = lambda word: self.word_num_map.get(word, len(self.VOCAB_SIZE))
        # 每一句诗都变成[[2, 28, 543, 104, 720, 1, 3], [2, 649, 48, 9, 2138, 1, 3], [2, 424, 4637, 2126, 1100, 1, 3]]
        poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]
        # 定义迭代次数
        self.max_step = len(poetrys_vector) // self.batch_size

        for i in range(self.max_step):

            start_index = i * self.batch_size
            end_index = start_index + self.batch_size

            # 取出每批次batch_size数量的诗
            batches = poetrys_vector[start_index:end_index]

            # 其中每批次中最长的诗
            length = max(map(len, batches))
            xdata = np.full((self.batch_size, length), self.word_num_map[' '], np.int32)

            # 每批次的诗的id赋值给xdata
            for row in range(self.batch_size):
                xdata[row, :len(batches[row])] = batches[row]
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]

            # xdata             ydata
            # [2,6,5,4,6,9,3]       [6,5,4,6,9,3]
            # [2,1,4,2,8,5,3]       [1,4,2,8,5,3]
            #
            # 序列的输入和输出（开头和结尾是特殊字符）

            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

        return None

    def rnn_network(self):
        """
        建立模型得出输出
        :return:
        """

        # 定义RNN结构类型  100个cell大于所有每个样本序列的数据长度
        cell_fun = tf.contrib.rnn.BasicRNNCell
        # rnn_size， RNN的单元数量100个
        cell = cell_fun(self.rnn_size)

        # 增加num_layers个重复RNN，多层RNN，这里指定2层cell
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

        # 初始化状态，形状[batch_size, rnn_size]
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        print("初始化所有隐层状态：", initial_state)

        with tf.variable_scope('rnnlm'):

            # 构建全连接层变量
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, len(self.VOCAB_SIZE) + 1])
            softmax_b = tf.get_variable("softmax_b", [len(self.VOCAB_SIZE) + 1])

            print("softmax_w:", softmax_w)
            print("softmax_b:", softmax_b)

            # 所有词的向量大小总词数，每个词的向量大小 [len(words)+1, 128]
            # 这里的embedding是需要训练的
            embedding = tf.get_variable("embedding", [len(self.VOCAB_SIZE) + 1, self.rnn_size])

            # 返回input_data中id在所有词embedding中的值
            # 输入层的维度为batch_size*num_steps,就变为[(batch_size), ? (num_steps), (rnn_size)]
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        print("词向量的形状：", inputs)  # [batch_size, ?, 100],1个批次，？个词，100维度

        # 一次运行多层RNN单元，outputs:为输出y的形状跟输入一样[1, ?, 128]，last_state:两个RNN隐层，每层都是[batch_size, 128]大小
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')

        # outputs: [64, ?], [64, ?]
        print("RNN 的输出和隐层状态:", outputs, last_state)

        output = tf.reshape(outputs, [-1, self.rnn_size])  # 每个词都是向量的长度[64, ?, 100维]

        print("RNN的输出层的形状改变：", output)

        # 将从RNN中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每一个时刻上都是一个长度为VOCAB_SIZE的数组
        logits = tf.matmul(output, softmax_w) + softmax_b    # [?, 6100]

        # 经过softmax层之后表示下一个位置是不同单词的概率，用于生成句子 # [?, 6100]
        probs = tf.nn.softmax(logits)

        return logits, last_state, probs, cell, initial_state

    def train(self, logits, last_state):

        targets = tf.reshape(self.output_targets, [-1])

        print("目标值：", targets)

        # 定义交叉熵损失，TensorFlow提供了sequence_loss_by_example函数来计算一个序列的交叉熵的和(由于每个样本数据是一个个的序列)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],  # 预测值 [batch * num_word, vocab_size]
            [targets],  # 期待的正确答案 [batch_size, num_steps]
            [tf.ones_like(targets, dtype=tf.float32)],
            len(self.VOCAB_SIZE))
        # 求平均损失
        cost = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(0.002).minimize(cost)

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 定义保存模型实例
            saver = tf.train.Saver(tf.all_variables())

            # 将所有的数据过50次
            for epoch in range(1):

                # 循环self.max_step批次
                for batche in range(30):
                    train_loss, _, _ = sess.run([cost, last_state, train_op],
                                                feed_dict={self.input_data: self.x_batches[batche], self.output_targets: self.y_batches[batche]})
                    print("第%d批次首诗，损失为%s" % (batche, train_loss))

                saver.save(sess, './tmp/model/poetry.module', global_step=epoch)

    def generate_poetry(self):

        def to_word(weights):
            """根据输出找到对应的词"""

            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))

            return self.VOCAB_SIZE[sample]

        _, last_state, probs, cell, initial_state = self.rnn_network()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, './tmp/model/poetry.module-0')

            state_ = sess.run(cell.zero_state(1, tf.float32))

            x = np.array([list(map(self.word_num_map.get, '['))])
            [probs_, state_] = sess.run([probs, last_state], feed_dict={self.input_data: x, initial_state: state_})

            # 根据概率求得单词
            word = to_word(probs_)

            # word = words[np.argmax(probs_)]
            poem = ''
            while word != ']':
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = self.word_num_map[word]
                [probs_, state_] = sess.run([probs, last_state], feed_dict={self.input_data: x, initial_state: state_})
                word = to_word(probs_)
            # word = words[np.argmax(probs_)]
            print(poem)


if __name__ == "__main__":
    rbp = RobotPoetry(batch_size=1)

    rbp.get_poetry()

    # # 返回预测结果，最后输出，概率分布，RNN结构，初始状态
    # logits, last_state, probs, cell, initial_state = rbp.rnn_network()
    #
    # rbp.train(logits, last_state)

    rbp.generate_poetry()