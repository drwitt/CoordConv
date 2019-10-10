
###############################################################################
###############################################################################
#COORDCONV Layers:
###############################################################################
###############################################################################

from tensorflow.python.layers import base
import tensorflow as tf

class AddCoords(base.Layer):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False, skiptile=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.skiptile = skiptile


    def call(self, input_tensor):
        """
        input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
        In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
        In the second case, skiptile, just concat
        """
        if not self.skiptile:
            input_tensor = tf.tile(input_tensor, [1, self.x_dim, self.y_dim, 1]) # (batch, 64, 64, 2)
            input_tensor = tf.cast(input_tensor, 'float32')

        batch_size_tensor = tf.shape(input_tensor)[0]  # get batch size

        xx_ones = tf.ones([batch_size_tensor, self.x_dim],
                          dtype=tf.int32)                       # e.g. (batch, 64)
        xx_ones = tf.expand_dims(xx_ones, -1)                   # e.g. (batch, 64, 1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0),
                            [batch_size_tensor, 1])             # e.g. (batch, 64)
        xx_range = tf.expand_dims(xx_range, 1)                  # e.g. (batch, 1, 64)


        xx_channel = tf.matmul(xx_ones, xx_range)               # e.g. (batch, 64, 64)
        xx_channel = tf.expand_dims(xx_channel, -1)             # e.g. (batch, 64, 64, 1)


        yy_ones = tf.ones([batch_size_tensor, self.y_dim],
                          dtype=tf.int32)                       # e.g. (batch, 64)
        yy_ones = tf.expand_dims(yy_ones, 1)                    # e.g. (batch, 1, 64)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0),
                            [batch_size_tensor, 1])             # (batch, 64)
        yy_range = tf.expand_dims(yy_range, -1)                 # e.g. (batch, 64, 1)

        yy_channel = tf.matmul(yy_range, yy_ones)               # e.g. (batch, 64, 64)
        yy_channel = tf.expand_dims(yy_channel, -1)             # e.g. (batch, 64, 64, 1)

        xx_channel = tf.cast(xx_channel, 'float32') / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, 'float32') / (self.y_dim - 1)
        xx_channel = xx_channel*2 - 1                           # [-1,1]
        yy_channel = yy_channel*2 - 1

        ret = tf.concat([input_tensor,
                         xx_channel,
                         yy_channel], axis=-1)    # e.g. (batch, 64, 64, c+2)

        if self.with_r:
            rr = tf.sqrt( tf.square(xx_channel)
                    + tf.square(yy_channel)
                    )
            ret = tf.concat([ret, rr], axis=-1)   # e.g. (batch, 64, 64, c+3)

        return ret

class CoordConv(base.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args,  **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim,
                                   y_dim=y_dim,
                                   with_r=with_r,
                                   skiptile=True)
        self.conv = tf.layers.Conv2D(*args, **kwargs)

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret

###############################################################################
###############################################################################
#COORDCONV CNN Models:
###############################################################################
###############################################################################

from tf_plus import BatchNormalization, Lambda   # BN + Lambda layers are custom, rest are just from tf.layers
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D
from tf_plus import he_normal, relu
from tf_plus import Layers, SequentialNetwork, l2reg
from general.tfutil import tf_reshape_like
from CoordConv import AddCoords, CoordConv

Deconv = tf.layers.Conv2DTranspose
ReLu = Lambda(lambda xx: relu(xx))
LReLu = Lambda(lambda xx: lrelu(xx))
Softmax = Lambda(lambda xx: tf.nn.softmax(xx, axis=-1))
Tanh = Lambda(lambda xx: tf.nn.tanh(xx))
GlobalPooling = Lambda(lambda xx: tf.reduce_mean(xx,[1,2]))

class DeconvPainter(Layers):
    '''A Deconv net that paints an image as directed by x,y coord inputs '''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1,
                 onthefly=True, use_mse_loss=False, use_sigm_loss=False):
        super(DeconvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.onthefly = onthefly
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss

        with tf.variable_scope("model"):
            self.l('model', SequentialNetwork([
                   Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                        # output_shape=[None, 2, 2, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 4, 4, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(64*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 8, 8, 64]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(32*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 16, 16, 32]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(32*mul, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 32, 32, 32]
                   BatchNormalization(momentum=0.9, epsilon=1e-5),
                   ReLu,
                   Deconv(1, (fs,fs), (2,2), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       # output_shape=[None, 64, 64, 1]
                   ], name='deconv_painter'))

        return

    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)

        input_coords = inputs[0]

        logits = self.model([input_coords])
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size

        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        if self.onthefly: # make labels on the fly
            indices = tf.reshape(tf.range(batch_size_tensor), [batch_size_tensor, 1])
            concat_indices = tf.concat([indices, input_coords], 1)
            labels_shape = [batch_size_tensor, self.x_dim, self.y_dim]
            labels = tf.sparse_to_dense(concat_indices, labels_shape, 1.)
            self.a('concat_indices', concat_indices)
        else:
            assert len(inputs) == 2, "Not on-the-fly, supply a target image"
            labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))
        # Validated -- the same as above
        #mse_loss_tf = tf.losses.mean_squared_error(labels=self.labels_flat, predictions=self.logits_flat)
        #self.a('mse_loss_tf', mse_loss_tf, trackable=True)

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection

        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return

class ConvImagePainter(Layers):
    '''A model net that paints a circle with one-hot center inputs'''

    def __init__(self, l2=0, fs=3, mul=1, use_sigm_loss=True,
                 use_mse_loss=False, version='working'):
        super(ConvImagePainter, self).__init__()
        self.use_sigm_loss = use_sigm_loss
        self.use_mse_loss = use_mse_loss
        assert version in ['simple', 'working', 'dilation'], "model version not supported"
        self.version = version

        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='model')
            self.l('model', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='model')
            self.l('model', net)

        return

    def call(self, inputs):

        assert len(inputs) == 2, "model requires 2 tensors: input_1hot, input_images"
        input_1hot, labels = inputs[0], inputs[1]

        logits = self.model(input_1hot)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        # Shapes:
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        labels_flat = Flatten()(labels)

        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool),
                            tf.cast(self.labels_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)

        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            raise ValueError('use either sigmoid or mse loss')

        return

class ConvRegressor(Layers):
    '''A model net that paints a circle with one-hot center inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64,
                 mul=1, _type='conv_uniform'):
        self.type=_type

        super(ConvRegressor, self).__init__()
        include_r = False

        def coordconv_model():
        #from onehots to coordinate with coordinate augmentation at beginning

            return SequentialNetwork([
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=True), # (batch, 64, 64, 4 or 5)
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (1,1), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(8, (3,3), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                Conv2D(2, (3,3), padding='same',
                        kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                MaxPooling2D(pool_size=64,strides=64,padding='valid'),

                ])

        def big_conv2():

            return SequentialNetwork([
                Conv2D(16*mul, (5,5), padding='same',
                   kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(16*mul, (1,1), padding='same',strides=1,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,

                BatchNormalization(momentum=0.9, epsilon=1e-5),
                Conv2D(16*mul, (3,3), padding='same',strides=1,
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,

                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(16*mul, (3,3), padding='same',
                       kernel_initializer=he_normal, strides=2,kernel_regularizer=l2reg(l2)),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (3,3), padding='same',strides=2,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(16*mul, (1,1), padding='same',strides=1,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(16*mul, (3,3), padding='same',strides=2,
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(2, (3,3), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                GlobalPooling #pool_size=-1,strides=1,padding='valid'),
                ])

        def big_conv():

            return SequentialNetwork([
                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (3,3), padding='same',
                   kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (3,3), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                MaxPooling2D(pool_size=2,strides=2,padding='valid'),
                Conv2D(16*mul, (3,3), padding='same',
                       kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Flatten(),
                Dense(64),
                ReLu,
                Dense(2) #pool_size=-1,strides=1,padding='valid'),
                ])


        with tf.variable_scope("model"):
            if self.type=='conv_uniform':
                self.l('model',big_conv())
            elif self.type=='conv_quarant':
                self.l('model',big_conv2())
            elif self.type=='coordconv':
                self.l('model',coordconv_model())

        return

    def call(self, inputs):

        assert len(inputs) == 2, "model requires 2 tensors: input_images (onehots), target coordinates"
        input_images, labels = inputs[0], inputs[1]

        logits = self.model(input_images)
        logits_flat = Flatten()(logits)

        # Shapes:
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)
        labels_flat = tf.cast(Flatten()(labels),tf.float32)

        self.a('logits', logits_flat)
        #self.a('logits_flat', logits_flat)
        self.a('labels', labels_flat)
        #self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()

        return logits

    def make_losses_and_metrics(self):

        mse_loss = tf.reduce_mean(
                    tf.square(self.logits - self.labels))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        eucl_dist = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.to_float(tf.square((64.0*self.logits)-(64.0*self.labels))),-1)))
        self.a('eucl_dist', eucl_dist, trackable=True)
        #self.a('mse_dist', mse_loss, trackable=True)

        self.a('loss', mse_loss+reg_loss, trackable=True)
        return

class CoordConvPainter(Layers):
    '''A CoordConv that paints a pixel as directed by x,y coord inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, mul=1, include_r=False,
                 use_mse_loss=False, use_sigm_loss=False):
        super(CoordConvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss

        with tf.variable_scope("model"):
            self.l('coordconvprep', SequentialNetwork([
                Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=False), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep'))

            self.l('model', SequentialNetwork([
                Conv2D(32*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(32*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*mul, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(1, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name='coordconv_painter'))
        return

    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)

        input_coords = inputs[0]

        prepped_coords = self.coordconvprep(input_coords)
        self.a('prepped_coords', prepped_coords)

        logits = self.model(prepped_coords)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size

        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        assert len(inputs) == 2, "Not on-the-fly, supply a target image"
        labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection

        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return

class UpsampleConvPainter(Layers):
    '''A upsample+conv model that paints a pixel as directed by x,y coord inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1, coordconv=False,
                 include_r=False, use_mse_loss=False, use_sigm_loss=False):
        super(UpsampleConvPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss

        with tf.variable_scope("model"):
            if coordconv:
                self.l('model', SequentialNetwork([
                       Lambda(lambda xx: tf.cast(xx, 'float32')),
                       Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                       UpSampling2D(size=(2,2)),
                           # output_shape=[None, 2, 2, 2]
                       CoordConv(2, 2, False, 64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)
                           ),
                           # output_shape=[None, 2, 2, 4]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 2, 2, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 4, 4, 64]
                       AddCoords(x_dim=4, y_dim=4, with_r=include_r, skiptile=True),
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 4, 4, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 8, 8, 64]
                       AddCoords(x_dim=8, y_dim=8, with_r=include_r, skiptile=True),
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 8, 8, 64]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 16, 16, 64]
                       AddCoords(x_dim=16, y_dim=16, with_r=include_r, skiptile=True),
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 16, 16, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 32, 32, 32]
                       AddCoords(x_dim=32, y_dim=32, with_r=include_r, skiptile=True),
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 32, 32, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 64, 64, 32]
                       AddCoords(x_dim=64, y_dim=64, with_r=include_r, skiptile=True),
                       Conv2D(1, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 64, 64, 1]
                       ], name='upsample_coordconv_painter'))
            else:
                self.l('model', SequentialNetwork([
                       Lambda(lambda xx: tf.cast(xx, 'float32')),
                       Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                       UpSampling2D(size=(2,2)),
                           # output_shape=[None, 2, 2, 2]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 2, 2, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 4, 4, 64]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                           # output_shape=[None, 4, 4, 64]
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 8, 8, 64]
                       Conv2D(64*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 8, 8, 64]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 16, 16, 64]
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 16, 16, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 32, 32, 32]
                       Conv2D(32*mul, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 32, 32, 32]
                       BatchNormalization(momentum=0.9, epsilon=1e-5),
                       ReLu,
                       UpSampling2D(size=(2,2)),  # output_shape=[None, 64, 64, 32]
                       Conv2D(1, (fs,fs), padding='same',
                           kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                           # output_shape=[None, 64, 64, 1]
                       ], name='upsample_conv_painter'))
        return

    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)

        input_coords = inputs[0]

        logits = self.model([input_coords])
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        batch_size_tensor = tf.shape(input_coords)[0]  # get batch size

        prob_flat = tf.nn.softmax(logits_flat)
        prob = tf_reshape_like(prob_flat, logits, name='softmax_prob')

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        assert len(inputs) == 2, "Not on-the-fly, supply a target image"
        labels = inputs[1]

        labels_flat = Flatten()(labels)

        self.a('prob', prob)
        self.a('prob_flat', prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('labels', labels)
        self.a('labels_flat', labels_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        xe_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.labels_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.labels_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.prob_flat, 1)   # index in [0,64*64)
        # convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim

        argmax_label = tf.argmax(self.labels_flat, 1)

        argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
        argmax_y_l = tf.mod(argmax_label, self.y_dim)

        self.a('argmax_prob', argmax_prob)
        self.a('argmax_label', argmax_label)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)
        self.a('argmax_x_l', argmax_x_l)
        self.a('argmax_y_l', argmax_y_l)


        correct = tf.equal(argmax_prob, argmax_label)
        accuracy = tf.reduce_mean(tf.to_float(correct))
        eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
        manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))

        self.a('reg_losses', reg_losses)
        self.a('correct', correct)
        self.a('accuracy', accuracy, trackable=True)
        self.a('eucl_dist', eucl_dist, trackable=True)
        self.a('manh_dist', manh_dist, trackable=True)
        self.a('xe_loss', xe_loss, trackable=True)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.labels_flat), 1) # num of pixels in intersection

        n_union = tf.reduce_sum(tf.to_float(tf.logical_or(tf.cast(painted, tf.bool), tf.cast(self.labels_flat, tf.bool))), 1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            self.a('loss', mse_loss+reg_loss, trackable=True)
        elif self.use_sigm_loss:
            self.a('loss', sigm_loss+reg_loss, trackable=True)
        else:
            self.a('loss', xe_loss+reg_loss, trackable=True)

        return



class CoordConvImagePainter(Layers):
    '''A CoordConv that paints an image  as directed by x,y coord inputs'''

    def __init__(self, l2=0, fs=3, x_dim=64, y_dim=64, mul=1, include_r=False,
            use_mse_loss=False, use_sigm_loss=True, interm_loss=None,
            no_softmax=False, version='working'):
        super(CoordConvImagePainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        self.interm_loss = interm_loss

        self.l('coordconvprep', SequentialNetwork([
            Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
            AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=include_r, skiptile=False), # (batch, 64, 64, 4 or 5)
            ], name='coordconvprep'))

        self.l('coordconvmodel', SequentialNetwork([
            Conv2D(32, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ReLu,
            Conv2D(32, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ReLu,
            Conv2D(64, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ReLu,
            Conv2D(64, (1,1), padding='valid',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ReLu,
            Conv2D(1, (1,1), padding='same',
                kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
            ], name='coordconvmodel'))

        self.l('sharpen', SequentialNetwork([
            Flatten(), # (batch, -1)
            Softmax,
            Lambda(lambda xx: tf.reshape(xx, [-1, x_dim, y_dim, 1])),
            ], name='sharpen'))

        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='convmodel')
            self.l('convmodel', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='convmodel')
            self.l('convmodel', net)

        if no_softmax:
            self.l('model', SequentialNetwork([
                ('coordconvprep', self.coordconvprep),
                ('coordconvmodel', self.coordconvmodel),
                ('convmodel', self.convmodel)
                ], name='model'))
        else:
            self.l('model', SequentialNetwork([
                ('coordconvprep', self.coordconvprep),
                ('coordconvmodel', self.coordconvmodel),
                ('sharpen', self.sharpen),
                ('convmodel', self.convmodel)
                ], name='model'))

        return

    def call(self, inputs):
        # Shapes:
        # input_coords: (batch, 2)
        # logits: (batch, x_dim, y_dim, 1)
        # logits_flat: (batch, x_dim*y_dim)

        if len(inputs) == 2:
            input_coords, input_images = inputs[0], inputs[1]
        elif len(inputs) == 3:
            input_coords, input_1hot, input_images = inputs[0], inputs[1], inputs[2]
        else:
            raise ValueError('model requires either 2 or 3 tensors: input_coords, (input_1hot,) input_images')

        prepped_coords = self.coordconvprep(input_coords)
        center_logits = self.coordconvmodel(prepped_coords)
        center_logits = tf.identity(center_logits, name='center_logits') # just to rename it
        center_logits_flat = Flatten()(center_logits)
        sharpened_logits = self.sharpen(center_logits)
        sharpened_logits = tf.identity(sharpened_logits, name='sharpened_logits') # just to rename it

        self.a('prepped_coords', prepped_coords)
        self.a('center_logits', center_logits)
        self.a('center_logits_flat', center_logits_flat)
        self.a('sharpened_logits', sharpened_logits)

        logits = self.model(input_coords)
        #logits = self.convmodel(input_1hot) # HACK to see if second part of the model works
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        center_prob = tf.identity(sharpened_logits, name='prob') # just to rename it
        center_prob_flat = Flatten()(center_prob)

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        images_flat = Flatten()(input_images)

        self.a('center_prob', center_prob)
        self.a('center_prob_flat', center_prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('images_flat', images_flat)

        if 'input_1hot' in locals():
            onehot_flat = Flatten()(input_1hot)
            self.a('onehot_flat', onehot_flat)

        self.make_losses_and_metrics()
        return logits

    def make_losses_and_metrics(self):

        # intermediate loss
        if hasattr(self, 'onehot_flat'):
            interm_softmax_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.center_logits_flat, labels=self.onehot_flat))
            interm_mse_loss = tf.reduce_mean(
                    tf.pow(self.center_logits_flat - self.onehot_flat, 2))
            self.a('interm_softmax_loss', interm_softmax_loss, trackable=True)
            self.a('interm_mse_loss', interm_mse_loss, trackable=True)

        # losses that have to do with only final images
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.images_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.images_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.center_prob_flat, 1)   # index in [0,64*64)
        ## convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim
        self.a('argmax_prob', argmax_prob)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)

        if hasattr(self, 'onehot_flat'):
            argmax_label = tf.argmax(self.onehot_flat, 1)
            argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
            argmax_y_l = tf.mod(argmax_label, self.y_dim)
            self.a('argmax_label', argmax_label)
            self.a('argmax_x_l', argmax_x_l)
            self.a('argmax_y_l', argmax_y_l)

            correct = tf.equal(argmax_prob, argmax_label)
            self.a('correct', correct)

            accuracy = tf.reduce_mean(tf.to_float(correct))
            eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
            manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))
            self.a('accuracy', accuracy, trackable=True)
            self.a('eucl_dist', eucl_dist, trackable=True)
            self.a('manh_dist', manh_dist, trackable=True)

        self.a('reg_losses', reg_losses)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.images_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool),
                            tf.cast(self.images_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            loss = mse_loss+reg_loss
        elif self.use_sigm_loss:
            loss = sigm_loss+reg_loss
        else:
            raise ValueError('use either sigmoid or mse loss')

        if self.interm_loss is not None:
            if self.interm_loss == 'mse':
                loss += interm_mse_loss
            elif self.interm_loss == 'softmax':
                loss += interm_softmax_loss
            else:
                raise ValueError('Support only `mse` or `softmax` intermediate loss')

        self.a('loss', loss, trackable=True)

        return

class DeconvBottleneckPainter(Layers):
    '''
    Like DeconvPainter but squeeze to 1 channel in between.
    Option to either sharpen that channel or not,
    with enforced loss or with softmax layer
    '''

    def __init__(self, l2=0, x_dim=64, y_dim=64, fs=3, mul=1,
                 use_mse_loss=False, use_sigm_loss=False,
                 interm_loss=None, no_softmax=False,
                 version='working'):
        super(DeconvBottleneckPainter, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_mse_loss = use_mse_loss
        self.use_sigm_loss = use_sigm_loss
        self.interm_loss = interm_loss


        net = build_deconv_coords2image(l2, mul, fs, name='coords2center')
        self.l('coords2center', net)

        self.l('sharpen', SequentialNetwork([
                Flatten(), # (batch, -1)
                Softmax,
                Lambda(lambda xx: tf.reshape(xx, [-1, x_dim, y_dim, 1])),
                ], name="sharpen"))

        if version == 'simple':
            net = build_simple_one_channel_onehot2image(l2, name='center2image')
            self.l('center2image', net)

        elif version == 'working':
            net = build_working_conv_onehot2image(l2, mul, fs, name='center2image')
            self.l('center2image', net)

            if no_softmax:
                self.l('model', SequentialNetwork([
                        ('coords2center', self.coords2center),
                        ('center2image', self.center2image)
                        ], name='model'))
            else:
                self.l('model', SequentialNetwork([
                        ('coords2center', self.coords2center),
                        ('sharpen', self.sharpen),
                        ('center2image', self.center2image)
                        ], name='model'))

        return

    def call(self, inputs):

        if len(inputs) == 2:
            input_coords, input_images = inputs[0], inputs[1]
        elif len(inputs) == 3:
            input_coords, input_1hot, input_images = inputs[0], inputs[1], inputs[2]
        else:
            raise ValueError('model requires either 2 or 3 tensors: input_coords, (input_1hot,) input_images')

        center_logits = self.coords2center(input_coords)
        center_logits = tf.identity(center_logits, name='center_logits') # just to rename it
        center_logits_flat = Flatten()(center_logits)
        sharpened_logits = self.sharpen(center_logits)
        sharpened_logits = tf.identity(sharpened_logits, name='sharpened_logits') # just to rename it

        self.a('center_logits', center_logits)
        self.a('center_logits_flat', center_logits_flat)
        self.a('sharpened_logits', sharpened_logits)


        logits = self.model(input_coords)
        logits = tf.identity(logits, name='logits') # just to rename it

        logits_flat = Flatten()(logits)

        center_prob = tf.identity(sharpened_logits, name='prob') # just to rename it
        center_prob_flat = Flatten()(center_prob)

        pixelwise_prob_flat = tf.nn.sigmoid(logits_flat)
        pixelwise_prob = tf_reshape_like(pixelwise_prob_flat, logits, name='pixelwise_prob')

        images_flat = Flatten()(input_images)

        self.a('center_prob', center_prob)
        self.a('center_prob_flat', center_prob_flat)
        self.a('pixelwise_prob', pixelwise_prob)
        self.a('pixelwise_prob_flat', pixelwise_prob_flat)
        self.a('logits', logits)
        self.a('logits_flat', logits_flat)
        self.a('images_flat', images_flat)

        if 'input_1hot' in locals():
            onehot_flat = Flatten()(input_1hot)
            self.a('onehot_flat', onehot_flat)

        self.make_losses_and_metrics()

        return logits

    def make_losses_and_metrics(self):
        # intermediate loss
        if hasattr(self, 'onehot_flat'):
            interm_softmax_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.center_logits_flat, labels=self.onehot_flat))
            interm_mse_loss = tf.reduce_mean(
                    tf.pow(self.center_logits_flat - self.onehot_flat, 2))
            self.a('interm_softmax_loss', interm_softmax_loss, trackable=True)
            self.a('interm_mse_loss', interm_mse_loss, trackable=True)

        # losses that have to do with only final images
        mse_loss = tf.reduce_mean(
                    tf.pow(self.logits_flat - self.images_flat, 2))

        sigm_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_flat, labels=self.images_flat))

        # regularizers
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()

        argmax_prob = tf.argmax(self.center_prob_flat, 1)   # index in [0,64*64)
        ## convert indices to 2D coordinates
        argmax_x = argmax_prob // self.x_dim
        argmax_y = argmax_prob % self.x_dim
        self.a('argmax_prob', argmax_prob)
        self.a('argmax_x', argmax_x)
        self.a('argmax_y', argmax_y)

        if hasattr(self, 'onehot_flat'):
            argmax_label = tf.argmax(self.onehot_flat, 1)
            argmax_x_l = tf.floordiv(argmax_label, self.y_dim)
            argmax_y_l = tf.mod(argmax_label, self.y_dim)
            self.a('argmax_label', argmax_label)
            self.a('argmax_x_l', argmax_x_l)
            self.a('argmax_y_l', argmax_y_l)

            correct = tf.equal(argmax_prob, argmax_label)
            self.a('correct', correct)

            accuracy = tf.reduce_mean(tf.to_float(correct))
            eucl_dist = tf.reduce_mean(tf.sqrt(tf.to_float(tf.square(argmax_x-argmax_x_l) + tf.square(argmax_y-argmax_y_l))))
            manh_dist = tf.reduce_mean(tf.to_float(tf.abs(argmax_x-argmax_x_l) + tf.abs(argmax_y-argmax_y_l)))
            self.a('accuracy', accuracy, trackable=True)
            self.a('eucl_dist', eucl_dist, trackable=True)
            self.a('manh_dist', manh_dist, trackable=True)

        self.a('reg_losses', reg_losses)
        self.a('mse_loss', mse_loss, trackable=True)
        self.a('sigm_loss', sigm_loss, trackable=True)
        self.a('reg_loss', reg_loss, trackable=True)

        # calculate IOU, useful if input_label is an image

        painted = tf.round(self.pixelwise_prob_flat)  # 1 if > 0.5, 0 if <= 0.5
        n_painted = tf.reduce_mean(tf.reduce_sum(painted, 1))

        n_intersection = tf.reduce_sum(tf.multiply(painted, self.images_flat), 1) # num of pixels in intersection
        n_union = tf.reduce_sum(
                tf.to_float(tf.logical_or(tf.cast(painted, tf.bool),
                            tf.cast(self.images_flat, tf.bool))),
                1)

        self.a('painted', painted)
        self.a('n_intersection', n_intersection)
        self.a('n_union', n_union)

        intersect_pixs = tf.reduce_mean(n_intersection)
        iou = tf.reduce_mean(tf.div(n_intersection, n_union))

        self.a('painted_pixs', n_painted, trackable=True)
        self.a('intersect_pixs', intersect_pixs, trackable=True)
        self.a('iou', iou, trackable=True)


        if self.use_mse_loss:
            loss = mse_loss+reg_loss
        elif self.use_sigm_loss:
            loss = sigm_loss+reg_loss
        else:
            raise ValueError('use either sigmoid or mse loss')

        if self.interm_loss is not None:
            if self.interm_loss == 'mse':
                loss += interm_mse_loss
            elif self.interm_loss == 'softmax':
                loss += interm_softmax_loss
            else:
                raise ValueError('Support only `mse` or `softmax` intermediate loss')

        self.a('loss', loss, trackable=True)
        return

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def build_working_conv_onehot2image(l2, mul, fs, name=''):
    net = SequentialNetwork([
                Conv2D(8*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(8*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(16*mul, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                #BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Conv2D(1, (fs,fs), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name=name)
    return net


class CoordConvGAN(Layers):
    '''basically DCGAN64 structure, enhanced with coordconv-like coordinate inputs'''

    def __init__(self, l2=0, x_dim=64, y_dim=64, cout=1,
                 add_r=False, coords_in_g=False, coords_in_d=False):

        super(CoordConvGAN, self).__init__()
        #self.l2 = l2
        #self.x_dim = x_dim
        #self.y_dim = y_dim
        #self.add_r = add_r
        self.coords_in_g = coords_in_g
        self.coords_in_d = coords_in_d

        with tf.variable_scope("generator"):
            self.l('coordconvprep_g', SequentialNetwork([
                Lambda(lambda xx: tf.expand_dims(tf.expand_dims(xx,1),1)),  # (batch, z_dim) -> (batch, 1, 1, z_dim)
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=add_r, skiptile=False), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep_g'))

            self.l('coordconv_generator', SequentialNetwork([
                Conv2D(8*64, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*4, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*4, (1,1), padding='valid',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*2, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*1, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*1, (3,3), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(64*1, (3,3), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ReLu,
                Conv2D(cout, (1,1), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                ], name='coordconv_generator'))

            self.l('deconv_generator', SequentialNetwork([
                Dense(4*4*8*64),
                Lambda(lambda xx: tf.reshape(xx, [-1, 4, 4, 8*64])),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*4, (5,5), (2,2), padding='same'),          # output_shape=[None, 8, 8, 512]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*2, (5,5), (2,2), padding='same'),          # output_shape=[None, 16, 16, 256]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*1, (5,5), (2,2), padding='same'),          # output_shape=[None, 32, 32, 128]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(cout, (5,5), (2,2), padding='same'),            # output_shape=[None, 64, 64, 3]
                Tanh
                ], name='deconv_generator'))

        with tf.variable_scope("discriminator"):

            self.l('coordconvprep_d', SequentialNetwork([
                AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=add_r, skiptile=True), # (batch, 64, 64, 4 or 5)
                ], name='coordconvprep_d'))

            self.l('discriminator_minus_last', SequentialNetwork([
                Conv2D(64, (5,5), 2, padding='same', activation=lrelu),
                Conv2D(128, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Conv2D(256, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Conv2D(512, (5,5), 2, padding='same'),
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                LReLu,
                Flatten(),
                #Dense(1)
                ], name='discriminator_minus_last'))


            self.l('discriminator', SequentialNetwork(
                [self.discriminator_minus_last,
                Dense(1)],
                name='discriminator'))

    def call(self, inputs, feature_matching_loss=False, feature_match_loss_weight=1.0):

        assert len(inputs) == 2+(feature_matching_loss), \
                'inputs: images, noises, (images2 if feature matching)'

        input_x, input_z = inputs[0], inputs[1]

        if feature_matching_loss:
            input_x2 = inputs[2]

        #self.build_model()

        # call generator
        if self.coords_in_g:
            enhanced_z = self.coordconvprep_g(input_z)
            g_out = self.coordconv_generator(enhanced_z)
            self.a('enhanced_z', enhanced_z)
        else:
            g_out = self.deconv_generator(input_z)

        self.a('g_out', g_out)

        if self.coords_in_g:
            self.a('fake_images', Tanh(g_out))
        else:
            self.a('fake_images', g_out)


        # call discriminator
        if self.coords_in_d:
            enhanced_img = self.coordconvprep_d(input_x)
            enhanced_img_fake = self.coordconvprep_d(g_out)
            d_real_logits = self.discriminator(enhanced_img)
            d_fake_logits = self.discriminator(enhanced_img_fake)

            if feature_matching_loss:
                enhanced_img2 = self.coordconvprep_d(input_x2)
                d_real_features = self.discriminator_minus_last(enhanced_img2)
                d_fake_features = self.discriminator_minus_last(enhanced_img_fake)
        else:
            d_real_logits = self.discriminator(input_x)
            d_fake_logits = self.discriminator(g_out)
            if feature_matching_loss:
                d_real_features = self.discriminator_minus_last(input_x2)
                d_fake_features = self.discriminator_minus_last(g_out)

        self.a('d_fake_logits', d_fake_logits)
        self.a('d_real_logits', d_real_logits)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))

        self.a('d_loss_real', d_loss_real, trackable=True)
        self.a('d_loss_fake', d_loss_fake, trackable=True)

        # get loss for generator
        g_loss_basic = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))
        self.a('g_loss_basic', g_loss_basic, trackable=True)

        # regularization loss
        reg_loss = tf.losses.get_regularization_loss()
        reg_losses = tf.losses.get_regularization_losses()
        self.a('reg_losses', reg_losses)
        self.a('reg_loss', reg_loss, trackable=True)


        if feature_matching_loss:
            g_loss_feture_match = tf.reduce_mean(tf.abs(tf.subtract(d_real_features, d_fake_features)))
            self.a('g_loss_feture_match', g_loss_feture_match, trackable=True)

            self.a('g_loss', g_loss_feture_match * feature_match_loss_weight + g_loss_basic + reg_loss, trackable=True)
            self.a('d_loss', d_loss_real + d_loss_fake, trackable=True)

        else:
            self.a('g_loss', g_loss_basic + reg_loss, trackable=True)
            self.a('d_loss', d_loss_real + d_loss_fake, trackable=True)

        # correct rate for discriminator
        pred_correct_real = tf.greater(d_real_logits, tf.zeros_like(d_real_logits))
        correct_real = tf.reduce_mean(tf.to_float(pred_correct_real))

        pred_correct_fake = tf.less(d_fake_logits, tf.zeros_like(d_fake_logits))
        correct_fake = tf.reduce_mean(tf.to_float(pred_correct_fake))

        self.a('correct_real', correct_real, trackable=True)
        self.a('correct_fake', correct_fake, trackable=True)

        return g_out

def build_simple_one_channel_onehot2image(l2, name=''):
    net = SequentialNetwork([
                Conv2D(1, (9,9), padding='same',
                    #kernel_initializer=he_normal,
                    kernel_initializer=tf.ones_initializer(),
                    bias_initializer=tf.constant_initializer([0.]),
                    kernel_regularizer=l2reg(l2)),
                ], name=name)
    return net

def build_deconv_coords2image(l2, mul, fs, name=''):
    net = SequentialNetwork([
                Lambda(lambda xx: tf.cast(xx, 'float32')),
                Lambda(lambda xx: tf.reshape(xx, [-1, 1, 1, 2])),
                Deconv(64*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                     # output_shape=[None, 2, 2, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 4, 4, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(64*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 8, 8, 64]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(32*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 16, 16, 32]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(32*mul, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 32, 32, 32]
                BatchNormalization(momentum=0.9, epsilon=1e-5),
                ReLu,
                Deconv(1, (fs,fs), (2,2), padding='same',
                    kernel_initializer=he_normal, kernel_regularizer=l2reg(l2)),
                    # output_shape=[None, 64, 64, 1]
                ], name=name)
    return net

###############################################################################
###############################################################################
#TRAINING
###############################################################################
###############################################################################

import sys
import os
import gzip
import pickle as pickle
import numpy as np
import h5py
import itertools
import random
from IPython import embed
import colorama

from general.util import tic, toc, tic2, toc2, tic3, toc3, mkdir_p, WithTimer
from general.tfutil import (get_collection_intersection_summary, log_scalars,
                            sess_run_dict, summarize_weights, summarize_opt,
                            tf_assert_all_init, tf_get_uninitialized_variables,
                            add_grad_summaries, add_grads_and_vars_hist_summaries,
                            image_summaries_traintest)
from general.stats_buddy import StatsBuddy
from general.lr_policy import LRPolicyConstant, LRPolicyStep, LRPolicyValStep
from tf_plus import setup_session_and_seeds, learning_phase, print_trainable_warnings
# from model_builders import (DeconvPainter, ConvImagePainter, ConvRegressor,
#                             CoordConvPainter, CoordConvImagePainter,
#                             DeconvBottleneckPainter, UpsampleConvPainter)
from util import make_standard_parser, merge_dict_append, average_dict_values

# all choices of tasks/architectures, to be used for --arch

# Tasks reported in paper
# deconv_classification / coordconv_classification
#   input: coords
#   output:onehot image

# deconv_rendering / coordconv_rendering
#   input:  coords
#   output: full square image

# conv_regressor / coordconv_regressor
#   input: onehot image
#   output: coords

arch_choices = [  # tasks reported in  paper
    'deconv_classification',
    'deconv_rendering',
    'coordconv_classification',
    'coordconv_rendering',
    'conv_regressor',
    'coordconv_regressor',  # TODO add
    # addtiontal tasks
    'conv_onehot_image',
    'deconv_bottleneck',
    'upsample_conv_coords',
    'upsample_coordconv_coords']

lr_policy_choices = ('constant', 'step', 'valstep')
intermediate_loss_choices = (None, 'softmax', 'mse')

def main():
    parser = make_standard_parser(
        'Coordconv',
        arch_choices=arch_choices,
        skip_train=True,
        skip_val=True)
    # re-add train and val h5s as optional
    parser.add_argument('--data_h5', type=str,
                        default='./data/rectangle_4_uniform.h5',
                        help='data file in hdf5.')
    parser.add_argument('--x_dim', type=int, default=64,
                        help='x dimension of the output image')
    parser.add_argument('--y_dim', type=int, default=64,
                        help='y dimension of the output image')
    parser.add_argument('--lrpolicy', type=str, default='constant',
                        choices=lr_policy_choices, help='LR policy.')
    parser.add_argument('--lrstepratio', type=float,
                        default=.1, help='LR policy step ratio.')
    parser.add_argument('--lrmaxsteps', type=int, default=5,
                        help='LR policy step ratio.')
    parser.add_argument('--lrstepevery', type=int, default=50,
                        help='LR policy step ratio.')
    parser.add_argument('--filter_size', '-fs', type=int, default=3,
                        help='filter size in deconv network')
    parser.add_argument('--channel_mul', '-mul', type=int, default=2,
        help='Deconv model channel multiplier to make bigger models')
    parser.add_argument('--use_mse_loss', '-mse', action='store_true',
                        help='use mse loss instead of cross entropy')
    parser.add_argument('--use_sigm_loss', '-sig', action='store_true',
                        help='use sigmoid loss instead of cross entropy')
    parser.add_argument('--interm_loss', '-interm', default=None,
        choices=(None, 'softmax', 'mse'),
        help='add intermediate loss to end-to-end painter model')
    parser.add_argument('--no_softmax', '-nosfmx', action='store_true',
                        help='Remove softmax sharpening layer in model')

    args = parser.parse_args()

    if args.lrpolicy == 'step':
        lr_policy = LRPolicyStep(args)
    elif args.lrpolicy == 'valstep':
        lr_policy = LRPolicyValStep(args)
    else:
        lr_policy = LRPolicyConstant(args)

    minibatch_size = args.minibatch
    train_style, val_style = (
        '', '') if args.nocolor else (
        colorama.Fore.BLUE, colorama.Fore.MAGENTA)

    sess = setup_session_and_seeds(args.seed, assert_gpu=not args.cpu)

    # 0. Load data or generate data on the fly
    print('Loading data: {}'.format(args.data_h5))

    if args.arch in ['deconv_classification',
                     'coordconv_classification',
                     'upsample_conv_coords',
                     'upsample_coordconv_coords']:

        # option a: generate data on the fly
        #data = list(itertools.product(range(args.x_dim),range(args.y_dim)))
        # random.shuffle(data)

        #train_test_split = .8
        #val_reps = int(args.x_dim * args.x_dim * train_test_split) // minibatch_size
        #val_size = val_reps * minibatch_size
        #train_end = args.x_dim * args.x_dim - val_size
        #train_x, val_x = np.array(data[:train_end]).astype('int'), np.array(data[train_end:]).astype('int')
        #train_y, val_y = None, None
        #DATA_GEN_ON_THE_FLY = True

        # option b: load the data
        fd = h5py.File(args.data_h5, 'r')

        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_onehots'], dtype=float)  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_onehots'], dtype=float)  # shape (768, 64, 64, 1)
        DATA_GEN_ON_THE_FLY = False

        # number of image channels
        image_c = train_y.shape[-1] if train_y is not None and len(train_y.shape) == 4 else 1

    elif args.arch == 'conv_onehot_image':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(
            fd['train_onehots'],
            dtype=int)  # shape (2368, 64, 64, 1)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(
            fd['val_onehots'],
            dtype=float)  # shape (768, 64, 64, 1)
        val_y = np.array(fd['val_imagegray'], dtype=float) / \
            255.0  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    elif args.arch == 'deconv_rendering':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_imagegray'], dtype=float) / \
            255.0  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    elif args.arch == 'conv_regressor' or args.arch == 'coordconv_regressor':
        fd = h5py.File(args.data_h5, 'r')
        train_y = np.array(
            fd['train_normalized_locations'],
            dtype=float)  # shape (2368, 2)
        # /255.0 # shape (2368, 64, 64, 1)
        train_x = np.array(fd['train_onehots'], dtype=float)
        val_y = np.array(
            fd['val_normalized_locations'],
            dtype=float)  # shape (768, 2)
        val_x = np.array(
            fd['val_onehots'],
            dtype=float)  # shape (768, 64, 64, 1)

        image_c = train_x.shape[-1]

    elif args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
        fd = h5py.File(args.data_h5, 'r')
        train_x = np.array(fd['train_locations'], dtype=int)  # shape (2368, 2)
        train_y = np.array(fd['train_imagegray'],
                           dtype=float) / 255.0  # shape (2368, 64, 64, 1)
        val_x = np.array(fd['val_locations'], dtype=float)  # shape (768, 2)
        val_y = np.array(fd['val_imagegray'], dtype=float) / 255.0  # shape (768, 64, 64, 1)

        # add one-hot anyways to track accuracy etc. even if not used in loss
        train_onehot = np.array(
            fd['train_onehots'],
            dtype=int)  # shape (2368, 64, 64, 1)
        val_onehot = np.array(
            fd['val_onehots'],
            dtype=int)  # shape (768, 64, 64, 1)

        image_c = train_y.shape[-1]

    train_size = train_x.shape[0]
    val_size = val_x.shape[0]

    # 1. CREATE MODEL
    input_coords = tf.placeholder(
        shape=(None,2),
        dtype='float32',
        name='input_coords')  # cast later in model into float
    input_onehot = tf.placeholder(
        shape=(None, args.x_dim, args.y_dim, 1),
        dtype='float32',
        name='input_onehot')
    input_images = tf.placeholder(
        shape=(None, args.x_dim, args.y_dim, image_c),
        dtype='float32',
        name='input_images')

    if args.arch == 'deconv_classification':
        model = DeconvPainter(l2=args.l2, x_dim=args.x_dim, y_dim=args.y_dim,
                              fs=args.filter_size, mul=args.channel_mul,
                              onthefly=DATA_GEN_ON_THE_FLY,
                              use_mse_loss=args.use_mse_loss,
                              use_sigm_loss=args.use_sigm_loss)

        model.a('input_coords', input_coords)

        if not DATA_GEN_ON_THE_FLY:
            model.a('input_onehot', input_onehot)

        model([input_coords]) if DATA_GEN_ON_THE_FLY else model([input_coords, input_onehot])

    if args.arch == 'conv_regressor':
        regress_type = 'conv_uniform' if 'uniform' in args.data_h5 else 'conv_quarant'
        model = ConvRegressor(l2=args.l2, mul=args.channel_mul,
                              _type=regress_type)
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        # call model on inputs
        model([input_onehot, input_coords])

    if args.arch == 'coordconv_regressor':
        model = ConvRegressor(l2=args.l2, mul=args.channel_mul,
                              _type='coordconv')
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        # call model on inputs
        model([input_onehot, input_coords])

    if args.arch == 'conv_onehot_image':
        model = ConvImagePainter(l2=args.l2, fs=args.filter_size, mul=args.channel_mul,
            use_mse_loss=args.use_mse_loss, use_sigm_loss=args.use_sigm_loss,
            version='working')
            # version='simple') # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)
        # call model on inputs
        model([input_onehot, input_images])

    if args.arch == 'deconv_rendering':
        model = DeconvPainter(l2=args.l2, x_dim=args.x_dim, y_dim=args.y_dim,
                              fs=args.filter_size, mul=args.channel_mul,
                              onthefly=False,
                              use_mse_loss=args.use_mse_loss,
                              use_sigm_loss=args.use_sigm_loss)
        model.a('input_coords', input_coords)
        model.a('input_images', input_images)
        # call model on inputs
        model([input_coords, input_images])

    elif args.arch == 'coordconv_classification':
        model = CoordConvPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            include_r=False,
            mul=args.channel_mul,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss)

        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)

        model([input_coords, input_onehot])
        #raise Exception('Not implemented yet')

    elif args.arch == 'coordconv_rendering':
        model = CoordConvImagePainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            include_r=False,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            interm_loss=args.interm_loss,
            no_softmax=args.no_softmax,
            version='working')
        # version='simple') # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)

        # always input three things to calculate relevant metrics
        model([input_coords, input_onehot, input_images])
    elif args.arch == 'deconv_bottleneck':
        model = DeconvBottleneckPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            interm_loss=args.interm_loss,
            no_softmax=args.no_softmax,
            version='working')  # version='simple' to hack a 9x9 all-ones filter solution
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model.a('input_images', input_images)

        # always input three things to calculate relevant metrics
        model([input_coords, input_onehot, input_images])

    elif args.arch == 'upsample_conv_coords' or args.arch == 'upsample_coordconv_coords':
        _coordconv = True if args.arch == 'upsample_coordconv_coords' else False
        model = UpsampleConvPainter(
            l2=args.l2,
            x_dim=args.x_dim,
            y_dim=args.y_dim,
            mul=args.channel_mul,
            fs=args.filter_size,
            use_mse_loss=args.use_mse_loss,
            use_sigm_loss=args.use_sigm_loss,
            coordconv=_coordconv)
        model.a('input_coords', input_coords)
        model.a('input_onehot', input_onehot)
        model([input_coords, input_onehot])

    print('All model weights:')
    summarize_weights(model.trainable_weights)
    #print 'Model summary:'
    print('Another model summary:')
    model.summarize_named(prefix='  ')
    print_trainable_warnings(model)

    # 2. COMPUTE GRADS AND CREATE OPTIMIZER
    # a placeholder for dynamic learning rate
    input_lr = tf.placeholder(tf.float32, shape=[])
    if args.opt == 'sgd':
        opt = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        opt = tf.train.AdamOptimizer(input_lr, args.beta1, args.beta2)

    grads_and_vars = opt.compute_gradients(
        model.loss,
        model.trainable_weights,
        gate_gradients=tf.train.Optimizer.GATE_GRAPH)
    train_step = opt.apply_gradients(grads_and_vars)
    # added to train_ and param_ collections
    add_grads_and_vars_hist_summaries(grads_and_vars)

    summarize_opt(opt)
    print('LR Policy:', lr_policy)

    # add_grad_summaries(grads_and_vars)
    if not args.arch.endswith('regressor'):
        image_summaries_traintest(model.logits)

    if 'input_onehot' in model.named_keys():
        image_summaries_traintest(model.input_onehot)
    if 'input_images' in model.named_keys():
        image_summaries_traintest(model.input_images)
    if 'prob' in model.named_keys():
        image_summaries_traintest(model.prob)
    if 'center_prob' in model.named_keys():
        image_summaries_traintest(model.center_prob)
    if 'center_logits' in model.named_keys():
        image_summaries_traintest(model.center_logits)
    if 'pixelwise_prob' in model.named_keys():
        image_summaries_traintest(model.pixelwise_prob)
    if 'center_logits' in model.named_keys():
        image_summaries_traintest(model.center_logits)
    if 'sharpened_logits' in model.named_keys():
        image_summaries_traintest(model.sharpened_logits)

    # 3. OPTIONALLY SAVE OR LOAD VARIABLES (e.g. model params, model running
    # BN means, optimization momentum, ...) and then finalize initialization
    saver = tf.train.Saver(
        max_to_keep=None) if (
        args.output or args.load) else None
    if args.load:
        ckptfile, miscfile = args.load.split(':')
        # Restore values directly to graph
        saver.restore(sess, ckptfile)
        with gzip.open(miscfile) as ff:
            saved = pickle.load(ff)
            buddy = saved['buddy']
    else:
        buddy = StatsBuddy()

    buddy.tic()    # call if new run OR resumed run

    # Check if special layers are initialized right
    #last_layer_w = [var for var in tf.global_variables() if 'painting_layer/kernel:0' in var.name][0]
    #last_layer_b = [var for var in tf.global_variables() if 'painting_layer/bias:0' in var.name][0]

    # Initialize any missed vars (e.g. optimization momentum, ... if not
    # loaded from checkpoint)
    uninitialized_vars = tf_get_uninitialized_variables(sess)
    init_missed_vars = tf.variables_initializer(
        uninitialized_vars, 'init_missed_vars')
    sess.run(init_missed_vars)
    # Print warnings about any TF vs. Keras shape mismatches
    # warn_misaligned_shapes(model)
    # Make sure all variables, which are model variables, have been
    # initialized (e.g. model params and model running BN means)
    tf_assert_all_init(sess)
    # tf.global_variables_initializer().run()

    # 4. SETUP TENSORBOARD LOGGING with tf.summary.merge

    train_histogram_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_histogram')
    train_scalar_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_scalar')
    test_histogram_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_histogram')
    test_scalar_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_scalar')
    param_histogram_summaries = get_collection_intersection_summary(
        'param_collection', 'orig_histogram')
    train_image_summaries = get_collection_intersection_summary(
        'train_collection', 'orig_image')
    test_image_summaries = get_collection_intersection_summary(
        'test_collection', 'orig_image')

    writer = None
    if args.output:
        mkdir_p(args.output)
        writer = tf.summary.FileWriter(args.output, sess.graph)

    # 5. TRAIN

    train_iters = (train_size) // minibatch_size + \
        int(train_size % minibatch_size > 0)
    if not args.skipval:
        val_iters = (val_size) // minibatch_size + \
            int(val_size % minibatch_size > 0)

    if args.ipy:
        print('Embed: before train / val loop (Ctrl-D to continue)')
        embed()

    while buddy.epoch < args.epochs + 1:
        # How often to log data
        def do_log_params(ep, it, ii): return True
        def do_log_val(ep, it, ii): return True

        def do_log_train(
            ep,
            it,
            ii): return (
            it < train_iters and it & it -
            1 == 0 or it >= train_iters and it %
            train_iters == 0)  # Log on powers of two then every epoch

        # 0. Log params
        if args.output and do_log_params(
                buddy.epoch,
                buddy.train_iter,
                0) and param_histogram_summaries is not None:
            params_summary_str, = sess.run([param_histogram_summaries])
            writer.add_summary(params_summary_str, buddy.train_iter)

        # 1. Forward test on validation set
        if not args.skipval:
            feed_dict = {learning_phase(): 0}
            if 'input_coords' in model.named_keys():
                val_coords = val_y if args.arch.endswith(
                    'regressor') else val_x
                feed_dict.update({model.input_coords: val_coords})

            if 'input_onehot' in model.named_keys():
                # if 'val_onehot' not in locals():
                if not args.arch == 'coordconv_rendering' and not args.arch == 'deconv_bottleneck':
                    if args.arch == 'conv_onehot_image' or args.arch.endswith('regressor'):
                        val_onehot = val_x
                    else:
                        val_onehot = val_y
                feed_dict.update({
                    model.input_onehot: val_onehot,
                })
            if 'input_images' in model.named_keys():
                feed_dict.update({
                    model.input_images: val_images,
                })

            fetch_dict = model.trackable_dict()

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                if test_image_summaries is not None:
                    fetch_dict.update(
                        {'test_image_summaries': test_image_summaries})
                if test_scalar_summaries is not None:
                    fetch_dict.update(
                        {'test_scalar_summaries': test_scalar_summaries})
                if test_histogram_summaries is not None:
                    fetch_dict.update(
                        {'test_histogram_summaries': test_histogram_summaries})

            with WithTimer('sess.run val iter', quiet=not args.verbose):
                result_val = sess_run_dict(
                    sess, fetch_dict, feed_dict=feed_dict)

            buddy.note_list(
                model.trackable_names(), [
                    result_val[k] for k in model.trackable_names()], prefix='val_')
            print((
                '[%5d] [%2d/%2d] val: %s (%.3gs/i)' %
                (buddy.train_iter,
                 buddy.epoch,
                 args.epochs,
                 buddy.epoch_mean_pretty_re(
                     '^val_',
                     style=val_style),
                    toc2())))

            if args.output and do_log_val(buddy.epoch, buddy.train_iter, 0):
                log_scalars(
                    writer, buddy.train_iter, {
                        'mean_%s' %
                        name: value for name, value in buddy.epoch_mean_list_re('^val_')}, prefix='val')
                if test_image_summaries is not None:
                    image_summary_str = result_val['test_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                if test_scalar_summaries is not None:
                    scalar_summary_str = result_val['test_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if test_histogram_summaries is not None:
                    hist_summary_str = result_val['test_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)

        # 2. Possiby Snapshot, possibly quit
        if args.output and args.snapshot_to and args.snapshot_every:
            snap_intermed = args.snapshot_every > 0 and buddy.train_iter % args.snapshot_every == 0
            #snap_end = buddy.epoch == args.epochs
            snap_end = lr_policy.train_done(buddy)
            if snap_intermed or snap_end:
                # Snapshot network and buddy
                save_path = saver.save(
                    sess, '%s/%s_%04d.ckpt' %
                    (args.output, args.snapshot_to, buddy.epoch))
                print('snappshotted model to', save_path)
                with gzip.open('%s/%s_misc_%04d.pkl.gz' % (args.output, args.snapshot_to, buddy.epoch), 'w') as ff:
                    saved = {'buddy': buddy}
                    pickle.dump(saved, ff)
                # Snapshot evaluation data and metrics
                _, _ = evaluate_net(
                    args, buddy, model, train_size, train_x, train_y, val_x, val_y, fd, sess)

        lr = lr_policy.get_lr(buddy)

        if buddy.epoch == args.epochs:
            if args.ipy:
                print('Embed: at end of training (Ctrl-D to exit)')
                embed()
            break   # Extra pass at end: just report val stats and skip training

        print('********* at epoch %d, LR is %g' % (buddy.epoch, lr))

        # 3. Train on training set
        if args.shuffletrain:
            train_order = np.random.permutation(train_size)
        tic3()
        for ii in range(train_iters):
            tic2()
            start_idx = ii * minibatch_size
            end_idx = min(start_idx + minibatch_size, train_size)

            if args.shuffletrain:  # default true
                batch_x = train_x[sorted(
                    train_order[start_idx:end_idx].tolist())]
                if train_y is not None:
                    batch_y = train_y[sorted(
                        train_order[start_idx:end_idx].tolist())]
                # if 'train_onehot' in locals():
                if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
                    batch_onehot = train_onehot[sorted(
                        train_order[start_idx:end_idx].tolist())]
            else:
                batch_x = train_x[start_idx:end_idx]
                if train_y is not None:
                    batch_y = train_y[start_idx:end_idx]
                # if 'train_onehot' in locals():
                if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
                    batch_onehot = train_onehot[start_idx:end_idx]

            feed_dict = {learning_phase(): 1, input_lr: lr}
            if 'input_coords' in model.named_keys():
                batch_coords = batch_y if args.arch.endswith(
                    'regressor') else batch_x
                feed_dict.update({model.input_coords: batch_coords})
            if 'input_onehot' in model.named_keys():
                # if 'batch_onehot' not in locals():
                # if not (args.arch == 'coordconv_rendering' and
                # args.add_interm_loss):
                if not args.arch == 'coordconv_rendering' and not args.arch == 'deconv_bottleneck':
                    if args.arch == 'conv_onehot_image' or args.arch.endswith(
                            'regressor'):
                        batch_onehot = batch_x
                    else:
                        batch_onehot = batch_y
                feed_dict.update({
                    model.input_onehot: batch_onehot,
                })
            if 'input_images' in model.named_keys():
                feed_dict.update({
                    model.input_images: batch_images,
                })

            fetch_dict = model.trackable_and_update_dict()

            fetch_dict.update({'train_step': train_step})

            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    fetch_dict.update(
                        {'train_histogram_summaries': train_histogram_summaries})
                if train_scalar_summaries is not None:
                    fetch_dict.update(
                        {'train_scalar_summaries': train_scalar_summaries})
                if train_image_summaries is not None:
                    fetch_dict.update(
                        {'train_image_summaries': train_image_summaries})

            with WithTimer('sess.run train iter', quiet=not args.verbose):
                result_train = sess_run_dict(
                    sess, fetch_dict, feed_dict=feed_dict)

            buddy.note_weighted_list(
                batch_x.shape[0], model.trackable_names(), [
                    result_train[k] for k in model.trackable_names()], prefix='train_')

            if do_log_train(buddy.epoch, buddy.train_iter, ii):
                print((
                    '[%5d] [%2d/%2d] train: %s (%.3gs/i)' %
                    (buddy.train_iter,
                     buddy.epoch,
                     args.epochs,
                     buddy.epoch_mean_pretty_re(
                         '^train_',
                         style=train_style),
                        toc2())))

            if args.output and do_log_train(buddy.epoch, buddy.train_iter, ii):
                if train_histogram_summaries is not None:
                    hist_summary_str = result_train['train_histogram_summaries']
                    writer.add_summary(hist_summary_str, buddy.train_iter)
                if train_scalar_summaries is not None:
                    scalar_summary_str = result_train['train_scalar_summaries']
                    writer.add_summary(scalar_summary_str, buddy.train_iter)
                if train_image_summaries is not None:
                    image_summary_str = result_train['train_image_summaries']
                    writer.add_summary(image_summary_str, buddy.train_iter)
                log_scalars(
                    writer, buddy.train_iter, {
                        'batch_%s' %
                        name: value for name, value in buddy.last_list_re('^train_')}, prefix='train')

            if ii > 0 and ii % 100 == 0:
                print('  %d: Average iteration time over last 100 train iters: %.3gs' % (
                    ii, toc3() / 100))
                tic3()

            buddy.inc_train_iter()   # after finished training a mini-batch

        buddy.inc_epoch()   # after finished training whole pass through set

        if args.output and do_log_train(buddy.epoch, buddy.train_iter, 0):
            log_scalars(
                writer, buddy.train_iter, {
                    'mean_%s' %
                    name: value for name, value in buddy.epoch_mean_list_re('^train_')}, prefix='train')

    print('\nFinal')
    print('%02d:%d val:   %s' % (buddy.epoch,
                                 buddy.train_iter,
                                 buddy.epoch_mean_pretty_re(
                                     '^val_',
                                     style=val_style)))
    print('%02d:%d train: %s' % (buddy.epoch,
                                 buddy.train_iter,
                                 buddy.epoch_mean_pretty_re(
                                     '^train_',
                                     style=train_style)))

    print('\nEnd of training. Saving evaluation results on whole train and val set.')

    final_tr_metrics, final_va_metrics = evaluate_net(
        args, buddy, model, train_size, train_x, train_y, val_x, val_y, fd, sess)

    print('\nFinal evaluation on whole train and val')
    for name, value in final_tr_metrics.items():
        print('final_stats_eval train_%s %g' % (name, value))
    for name, value in final_va_metrics.items():
        print('final_stats_eval val_%s %g' % (name, value))

    print('\nfinal_stats epochs %g' % buddy.epoch)
    print('final_stats iters %g' % buddy.train_iter)
    print('final_stats time %g' % buddy.toc())
    for name, value in buddy.epoch_mean_list_all():
        print('final_stats %s %g' % (name, value))

    if args.output:
        writer.close()   # Flush and close


def evaluate_net(args, buddy, model, train_size, train_x, train_y,
                 val_x, val_y, fd, sess, write_x=True, write_y=True):

    minibatch_size = args.minibatch
    train_iters = (train_size) // minibatch_size + \
        int(train_size % minibatch_size > 0)

    # 0 even for train set; because it's evalutation
    feed_dict_tr = {learning_phase(): 0}
    feed_dict_va = {learning_phase(): 0}

    if args.output:
        final_fetch = {'logits': model.logits}
        if 'prob' in model.named_keys():
            final_fetch.update({'prob': model.prob})
        if 'pixelwise_prob' in model.named_keys():
            final_fetch.update({'pixelwise_prob': model.pixelwise_prob})

        if args.arch == 'coordconv_rendering' or args.arch == 'deconv_bottleneck':
            final_fetch.update({
                'center_logits': model.center_logits,
                # 'sharpened_logits': model.sharpened_logits, # or center_prob
                'center_prob': model.center_prob,  # or center_prob
            })

        ff = h5py.File(
            '%s/evaluation_%04d.h5' %
            (args.output, buddy.epoch), 'w')

        # create dataset but write later
        for kk in list(final_fetch.keys()):
            if args.arch.endswith('regressor'):
                ff.create_dataset(kk + '_train', (minibatch_size, 2),
                    maxshape=(train_size, 2), dtype=float,
                    compression='lzf', chunks=True)
            else:
                ff.create_dataset(kk + '_train',
                    (minibatch_size, args.x_dim, args.y_dim, 1),
                    maxshape=(train_size, args.x_dim, args.y_dim, 1),
                    dtype=float, compression='lzf', chunks=True)

        # create dataset and write immediately
        if write_x:
            ff.create_dataset('inputs_val', data=val_x)
            ff.create_dataset('inputs_train', data=train_x)
        if write_y:
            ff.create_dataset('labels_val', data=val_y)
            ff.create_dataset('labels_train', data=train_y)

    for ii in range(train_iters):
        start_idx = ii * minibatch_size
        end_idx = min(start_idx + minibatch_size, train_size)

        if 'input_onehot' in model.named_keys():
            feed_dict_tr.update({model.input_onehot: np.array(
                fd['train_onehots'][start_idx:end_idx], dtype=float)})
            if ii == 0:
                feed_dict_va.update(
                    {model.input_onehot: np.array(fd['val_onehots'], dtype=float)})
                #feed_dict_va.update({model.input_onehot: val_onehot})
        if 'input_images' in model.named_keys():
            feed_dict_tr.update({model.input_images: np.array(
                fd['train_imagegray'][start_idx:end_idx], dtype=float) / 255.0})
            if ii == 0:
                feed_dict_va.update({model.input_images: np.array(
                    fd['val_imagegray'], dtype=float) / 255.0})
                #feed_dict_va.update({model.input_images: val_images})

        if 'input_coords' in model.named_keys():
            if args.arch.endswith('regressor'):
                _loc_keys = (
                    'train_normalized_locations',
                    'val_normalized_locations',
                    'float32')
            else:
                _loc_keys = (
                    'train_locations',
                    'val_locations',
                    'int32')
            feed_dict_tr.update({model.input_coords: np.array(
                fd[_loc_keys[0]][start_idx:end_idx], dtype=_loc_keys[2])})
            if ii == 0:
                feed_dict_va.update({model.input_coords: np.array(
                    fd[_loc_keys[1]], dtype=_loc_keys[2])})

        _final_tr_metrics = sess_run_dict(
            sess, model.trackable_dict(), feed_dict=feed_dict_tr)
        _final_tr_metrics['weights'] = end_idx - start_idx

        final_tr_metrics = _final_tr_metrics if ii == 0 else merge_dict_append(
            final_tr_metrics, _final_tr_metrics)

        if args.output:
            if ii == 0:  # do only once
                final_va = sess_run_dict(
                    sess, final_fetch, feed_dict=feed_dict_va)
                for kk in list(final_fetch.keys()):
                    ff.create_dataset(kk + '_val', data=final_va[kk])

            final_tr = sess_run_dict(sess, final_fetch, feed_dict=feed_dict_tr)
            for kk in list(final_fetch.keys()):
                if start_idx > 0:
                    n_samples_ = ff[kk + '_train'].shape[0]
                    ff[kk + '_train'].resize(n_samples_ +
                                             end_idx - start_idx, axis=0)
                ff[kk + '_train'][start_idx:, ...] = final_tr[kk]

    final_va_metrics = sess_run_dict(
        sess, model.trackable_dict(), feed_dict=feed_dict_va)
    final_tr_metrics = average_dict_values(final_tr_metrics)

    if args.output:
        with open('%s/evaluation_%04d_metrics.pkl' % (args.output, buddy.epoch), 'w') as ffmetrics:
            tosave = {'train': final_tr_metrics,
                      'val': final_va_metrics,
                      'time_elapsed': buddy.toc()
                      }
            pickle.dump(tosave, ffmetrics)

        ff.close()
    else:
        print('\nEpoch %d evaluation on whole train and val' % buddy.epoch)
        print('Time elapsed: {}'.format(buddy.toc()))
        for name, value in final_tr_metrics.items():
            print('final_stats_eval train_%s %g' % (name, value))
        for name, value in final_va_metrics.items():
            print('final_stats_eval val_%s %g' % (name, value))

    return final_tr_metrics, final_va_metrics


if __name__ == '__main__':
    main()
