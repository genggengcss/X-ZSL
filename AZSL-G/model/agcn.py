from layers import *
from metrics import *
from inits import *


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class Model_dense(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.coefs = None
        self.in_output = None
        self.W_c = None
        self.W = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.decay = 0

    def _build(self):
        raise NotImplementedError



    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()






        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

        # calculate gradient with respect to input, only for dense model
        self.grads = tf.gradients(self.loss, self.inputs)[0]  # does not work on sparse vector

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN_dense_mse(Model_dense):
    def __init__(self, args, placeholders, input_dim, node_num, **kwargs):
        super(GCN_dense_mse, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.args = args

        self.input_dim = input_dim
        self.node_num = node_num
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=placeholders['learning_rate'])

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=placeholders['learning_rate'])
        self.build()



    # training: unseen to seen, testing: seen to unseen
    def add_atten_cross(self, inputs):
        output_size = int(inputs.shape[-1])
        outputs_select = inputs

        # compute the cosine distance simlarity
        one = tf.ones([1, output_size], tf.float32)
        tff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(outputs_select), 1)), 1)
        norm = tf.matmul(tff, one)
        outputs_select_nor = tf.divide(outputs_select, norm)
        logits = tf.matmul(outputs_select_nor, tf.transpose(outputs_select_nor))

        train_mask = tf.cast(self.placeholders['labels_adj_mask'], dtype=tf.float32)
        train_mask /= tf.reduce_mean(train_mask)
        val_mask = tf.cast(self.placeholders['val_adj_mask'], dtype=tf.float32)
        val_mask /= tf.reduce_mean(val_mask)

        train_logits = logits * train_mask  # (3969, 3969)
        train_logits = train_logits * tf.transpose(val_mask)
        val_logits = logits * val_mask
        val_logits = val_logits * tf.transpose(train_mask)

        # self-attention
        diagonal_logits = tf.eye(self.node_num)
        diagonal_logits = tf.cast(diagonal_logits, dtype=tf.float32)
        diagonal_logits /= tf.reduce_mean(diagonal_logits)
        logits = train_logits + val_logits + diagonal_logits

        # attention weights
        coefs = tf.nn.softmax(logits)
        outputs_update = tf.matmul(coefs, inputs)  # (3969, 2048)
        return outputs_update, coefs



    def MLP(self, _x, _weights, _biases):
        layer1 = tf.nn.relu(tf.add(tf.matmul(_x, _weights['h1']), _biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2']))
        return tf.matmul(layer2, _weights['out']) + _biases['out']

    def add_atten_MLP(self, inputs):

        self.weights = {
            'h1': glorot([self.output_dim * 2, 2048]),
            'h2': glorot([2048, 512]),
            'out': glorot([512, 1])
        }

        self.baises = {
            'b1': zeros([2048]),
            'b2': zeros([512]),
            'out': zeros([1])
        }

        outputs = list()

        for i in range(len(self.support)):
            conce_batch = list()
            for j in range(len(self.support)):
                conce = tf.concat([inputs[i], inputs[j]], -1)
                conce_batch.append(conce)
            conce_batch = tf.convert_to_tensor(conce_batch)
            output_MLP = self.MLP(conce_batch, self.weights, self.baises)
            output_MLP = tf.squeeze(output_MLP, 1)
            outputs.append(output_MLP)

        outputs = tf.convert_to_tensor(outputs)  # weight matrix

        trainval_mask = tf.cast(self.placeholders['trainval_adj_mask'], dtype=tf.float32)
        trainval_mask /= tf.reduce_mean(trainval_mask)
        outputs = outputs * trainval_mask  # (3969, 3969) remove invalid classes
        logits = outputs * tf.transpose(trainval_mask)  # (3969, 3969) remove invalid weights


        #  only consider neighborhood, filter the unconnected edge
        # biase_mask = tf.cast(self.placeholders['biase_mask'], dtype=tf.float32)
        # logits = logits * biase_mask  # biase_mask: (3969, 3969)

        # attention weights
        coefs = tf.nn.softmax(logits)
        inputs_update = tf.matmul(coefs, inputs)  # (3969, 2048)
        # return outputs_update, coefs
        return inputs_update, coefs




    # attention: seen-unseen, compute the similarity with cosine distance
    def add_atten_su_cos(self, inputs):

        output_size = int(inputs.shape[-1])
        outputs_select = inputs

        # compute the cosine distance similarity
        one = tf.ones([1, output_size], tf.float32)
        tff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(outputs_select), 1)), 1)
        norm = tf.matmul(tff, one)
        outputs_select_nor = tf.divide(outputs_select, norm)
        logits = tf.matmul(outputs_select_nor, tf.transpose(outputs_select_nor))

        # # select training+testing nodes:
        trainval_mask = tf.cast(self.placeholders['trainval_adj_mask'], dtype=tf.float32)
        trainval_mask /= tf.reduce_mean(trainval_mask)
        logits = logits * trainval_mask  # (3969, 3969) remove invalid classes
        logits = logits * tf.transpose(trainval_mask)  # (3969, 3969) remove invalid weights



        #  only consider neighborhood, filter the unconnected edge
        # biase_mask = tf.cast(self.placeholders['biase_mask'], dtype=tf.float32)
        # logits = logits * biase_mask # biase_mask: (3969, 3969)

        # attention weights
        coefs = tf.nn.softmax(logits)
        outputs_update = tf.matmul(coefs, inputs)  # (3969, 2048)
        # return outputs_update, coefs
        return outputs_update, coefs


    # attention: seen-unseen, directly dot product to compute the similarity between seen and unseen classes
    def add_atten_su_dot(self, inputs):
        #### add attention layer start
        # input_size = int(inputs.shape[-1])
        # output_size = int(inputs.shape[-1])
        #
        # W = tf.Variable(tf.random_uniform([input_size, output_size], -0.005, 0.005))
        # b = tf.Variable(tf.random_uniform([output_size], -0.005, 0.005))
        #
        # outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W)+b
        # outputs = tf.reshape(outputs, \
        #                      tf.concat([tf.shape(inputs)[:-1], [output_size]], 0)
        #                      )

        # outputs = inputs
        #
        # # compute the similarity between seen nodes and unseen
        # logits = tf.matmul(outputs, tf.transpose(outputs))  # (3969,3969)
        # # select training+testing nodes:
        # trainval_mask = tf.cast(self.placeholders['trainval_adj_mask'], dtype=tf.float32)
        # # trainval_mask /= tf.reduce_mean(trainval_mask)
        # logits = logits * trainval_mask  # (3969, 2048)
        # logits = logits * tf.transpose(trainval_mask)  # (3969, 3969) remove invalid weights



        trainval_mask = tf.cast(self.placeholders['trainval_mask'], dtype=tf.float32)
        trainval_mask /= tf.reduce_mean(trainval_mask)
        outputs = inputs * trainval_mask

        # compute the similarity between seen nodes and unseen
        logits = tf.matmul(outputs, tf.transpose(outputs))  # (3969,3969)
        # select training+testing nodes:



        # attention weights
        coefs = tf.nn.softmax(logits)
        outputs_update = tf.matmul(coefs, inputs)  # (3969, 2048)
        return outputs_update, coefs




    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += self.args.weight_decay * tf.nn.l2_loss(var)

        # compute attention weights among the last layer outputs
        # self.outputs_atten, self.coefs = self.add_atten_su_cos(self.outputs)
        self.outputs_atten, self.coefs = self.add_atten_su_cos(self.outputs)

        # for var in self.weights.values():
        #     self.loss += self.args.weight_decay * tf.nn.l2_loss(var)
        # for var in self.baises.values():
        #     self.loss += self.args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += mask_mse_loss(self.outputs_atten, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                   self.placeholders['labels_mask'])
        # self.loss += mask_mse_loss(self.outputs, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
        #                            self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = mask_mse_loss(self.outputs_atten, tf.nn.l2_normalize(self.placeholders['labels'], dim=1),
                                      self.placeholders['labels_mask'])

    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.args.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,

                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden1,
                                            output_dim=self.args.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden2,
                                            output_dim=self.args.hidden3,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden3,
                                            output_dim=self.args.hidden4,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=False,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden4,
                                            output_dim=self.args.hidden5,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.maximum(x, 0.2 * x),
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden5,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: tf.nn.l2_normalize(x, dim=1),
                                            # act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])  # each output of hidden layer
            # self.in_output = hidden
            # hidden_atten = self.add_atten_su_cos(hidden)
            # self.activations.append(hidden_atten)
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

    def predict(self):
        return self.outputs_atten
