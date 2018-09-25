import tensorflow as tf

class SEN():
    def __init__(self, is_train):
        self.is_train = is_train
        self.bn_params = {'is_training': self.is_train,
                          'center': True, 'scale': True,
                          'updates_collections': None}
        self.build_model()

    def init_place_holder(self):
        self.x1 = tf.placeholder(tf.float32, shape=[None, None, 128, 1])
        self.x2 = tf.placeholder(tf.float32, shape=[None, None, 128, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

    def conv(self, inputs, filters, kernel, stride):
        return tf.contrib.layers.conv2d(inputs, filters, kernel, stride,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                        normalizer_params=self.bn_params)
    
    def fc(self, inputs, num_units, act=tf.nn.relu):
        keep_rate = 0.5
        _inputs = tf.contrib.layers.dropout(inputs, keep_rate, is_training=self.is_train)
        return tf.contrib.layers.fully_connected(_inputs, num_units,
                                                 activation_fn=act,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,
                                                 normalizer_params=self.bn_params)
    
    def cnn(self, inputs, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h = self.conv(inputs, 128, [4, 128], [1, 128])
            h = self.conv(h, 256, [4, 1], [1, 1])
            h = self.conv(h, 256, [4, 1], [1, 1])
        return h

    def reduce_var(self, inputs, axis):
        m = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        devs_squared = tf.square(inputs - m)
        return tf.reduce_mean(devs_squared, axis=axis)

    def build_model(self):
        # placeholder
        self.init_place_holder()
        
        # Early Conv.
        h1 = tf.squeeze(self.cnn(self.x1, 'cnn'), axis=2)
        h2 = tf.squeeze(self.cnn(self.x2, 'cnn', reuse=True), axis=2)

        # Consine Similarity
        num = tf.matmul(h1, h2, transpose_b=True)
        h1_norm = tf.sqrt(tf.reduce_sum(tf.square(h1), axis=2, keep_dims=True))
        h2_norm = tf.sqrt(tf.reduce_sum(tf.square(h2), axis=2, keep_dims=True))
        denom =  tf.matmul(h1_norm, h2_norm, transpose_b=True)
        fms = tf.expand_dims(tf.div(num, denom), 3)

        # Late Conv.
        h = self.conv(fms, 64, [3, 3], [1, 1])
        h = tf.layers.max_pooling2d(h, [3, 3], [3, 3], padding='same')
        h = self.conv(h, 128, [3, 3], [1, 1])
        h = tf.layers.max_pooling2d(h, [3, 3], [3, 3], padding='same')
        h = self.conv(h, 256, [3, 3], [1, 1])

        # Global Pooling
        g_max = tf.reduce_max(h, [1, 2])
        g_avg = tf.reduce_mean(h, [1, 2])
        g_var = self.reduce_var(h, [1, 2])
        h = tf.concat([g_max, g_avg, g_var], axis=1)

        # Classifier
        h = self.fc(h, 1024, act=tf.nn.relu)
        h = self.fc(h, 1024, act=tf.nn.relu)
        logits = self.fc(h, 2, act=None)
        self.predictions = tf.nn.softmax(logits)
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)
        self.loss = tf.reduce_mean(self.loss)

        # Train
        if self.is_train:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    def calculate(self, sess, batch):
        feed_dict = {self.x1: batch.x1,
                     self.x2: batch.x2,
                     self.y: batch.y}
        predictions = sess.run(self.predictions, feed_dict=feed_dict)
        return predictions