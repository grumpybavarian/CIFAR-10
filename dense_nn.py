import tensorflow as tf
import os

def dense_layer(layer_input, num_neurons, keep_prob):
    layer = tf.layers.dense(inputs=layer_input,
                            units=num_neurons,
                            activation=tf.nn.relu,
                            use_bias=True)
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    return layer

def unpickle(file):
    """
    Taken from http://www.cs.toronto.edu/~kriz/cifar.html
    :param file: 
    :return: 
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Model(object):
    def __init__(self, num_hidden_layers=8, num_neurons=1024):
        self.input_data = tf.placeholder(tf.float32, shape=(None, 32*32*3))
        self.labels = tf.placeholder(tf.int32, shape=(None,))
        self.y_ = tf.one_hot(self.labels, 10)
        self.network = self.input_data

        self.keep_prob = tf.placeholder(tf.float32)

        for i in range(num_hidden_layers):
            self.network = dense_layer(self.network, num_neurons=num_neurons, keep_prob=self.keep_prob)

        self.y = dense_layer(self.network, num_neurons=10, keep_prob=1.0)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_)
        self.train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def train(self):
        data_dir = "./data/"
        data_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        test_batch = "test_batch"

        num_epochs = 50

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):
                for batch in data_batches:
                    batch_dict = unpickle(os.path.join(data_dir, batch))
                    labels = batch_dict[b"labels"]
                    data = batch_dict[b"data"]

                    acc, _ = sess.run([self.accuracy, self.train_step], feed_dict={self.input_data: data,
                                                                                   self.labels: labels,
                                                                                   self.keep_prob: 0.5})

                    print(acc)

            test_batch_dict = unpickle(os.path.join(data_dir, test_batch))
            print(sess.run(self.train_step, feed_dict={self.input_data: test_batch_dict[b"data"],
                                                       self.labels: test_batch_dict[b"labels"],
                                                       self.keep_prob: 1.0}))





if __name__ == "__main__":
    m = Model()
    m.train()
