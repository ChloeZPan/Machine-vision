"""All the layer functions go here.

"""

from __future__ import print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the fully connected layer.
        b (np.array): the biases of the fully connected layer.
        shape (tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.
        name (str): the name of the layer.

    """

    def __init__(
        self, d_in, d_out, weights_init=None, bias_init=None, name="FullyConnected"
    ):
        shape = (d_out, d_in)
        self.W = weights_init.initialize(shape) \
            if weights_init else np.random.randn(*shape).astype(np.float32)
        self.b = bias_init.initialize((shape[0])) \
            if bias_init else np.random.randn(shape[0]).astype(np.float32)
        self.shape = shape
        self.name = name

    def __repr__(self):
        return "{}({}, {})".format(self.name, self.shape[0], self.shape[1])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        Y = np.dot(self.W, x) + self.b
        return Y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer.
            dv_y (np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input.
            dv_W (np.array): The derivative of the loss with respect to the
                weights.
            dv_b (np.array): The derivative of the loss with respect to the
                biases.

        """

        # TODO: write your implementation below
        dv_x = np.empty(x.shape, dtype=np.float32)
        dv_W = np.empty(self.W.shape, dtype=np.float32)
        dv_b = np.empty(self.b.shape, dtype=np.float32)

        # the derivative of the output with respect to the input, weight and bias
        dv_x = np.dot(self.W.T, dv_y)
        dv_W = np.multiply(dv_y.reshape(dv_y.shape[0], -1), x.reshape(x.shape[0], -1).T)
        dv_b = dv_y

        # don't change the order of return values
        return dv_x, dv_W, dv_b


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, filter_height, filter_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            in_channels).
        filter_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, filter_height, filter_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, in_channel, out_channel, kernel_size, stride, padding,
            weights_init=None, bias_init=None, name="Conv2D"):
        filter_size = (out_channel, in_channel, *kernel_size)

        self.W = weights_init.initialize(filter_size) \
            if weights_init else np.random.randn(*filter_size).astype(np.float32)
        self.b = bias_init.initialize((filter_size[0], 1)) \
            if bias_init else np.random.randn(out_channel, 1).astype(np.float32)

        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        y_shape = (
            self.W.shape[0],
            int((x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for k in range(y.shape[0]):
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    y[k, i, j] = np.sum(
                        x_padded[
                            :,
                            i * s[0] : i * s[0] + self.W.shape[2],
                            j * s[1] : j * s[1] + self.W.shape[3]
                        ] * self.W[k]
                    ) + self.b[k]
        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).
            dv_y (np.array): The derivative of the loss with respect to the
                output. A 3D array of shape (out_channels, out_heights,
                out_weights).

        Returns:
            dv_x (np.array): The derivative of the loss with respect to the
                input. It has the same shape as x.
            dv_W (np.array): The derivative of the loss with respect to the
                weights. It has the same shape as self.W
            dv_b (np.array): The derivative of the loss with respect to the
                biases. It has the same shape as self.b

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # TODO: write your implementation below

        dv_W = np.empty(self.W.shape, dtype=np.float32)
        dv_b = np.empty(self.b.shape, dtype=np.float32)
        dv_x = np.empty(x.shape, dtype=np.float32)

        in_c, in_h, in_w = x.shape
        out_c, in_c, f_h, f_w = self.W.shape
        out_c, out_h, out_w = dv_y.shape
        h_stride = s[0]
        w_stride = s[1]
        h_pad = p[0]
        w_pad = p[1]


        # for f in range(out_c):
        #     for i in range(f_h):
        #         for j in range(f_w):
        #             for k in range(out_h):
        #                 for l in range(out_w):
        #                     for c in range(in_c):
        #                         dv_W[f, c, i, j] += x_padded[c, i*h_stride+k, j*w_stride+l]*dv_y[f, k, l]
        #
        # doutp = np.pad(dv_y, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')
        # dv_x = np.pad(dv_x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')
        # w_ = np.zeros_like(self.W)
        # for i in range(f_h):
        #     for j in range(f_w):
        #         w_[:, :, i, j] = self.W[:, :, f_h - i - 1, f_w - j - 1]
        #
        # for f in range(out_c):
        #     for i in range(f_h+2*h_pad):
        #         for j in range(f_w+2*w_pad):
        #             for k in range(out_h):
        #                 for l in range(out_w):
        #                     for c in range(in_c):
        #                         dv_x[c, i, j] += doutp[f, i+k, j+l]*w_[c, k, l]
        # dv_x = dv_x[:, :, h_pad:-h_pad, w_pad:-w_pad]
        # dv_b = np.sum(dv_y, axis=(0, 2, 3))
        for c in range(out_c):
            for f in range(in_c):
                for h in range(out_h):
                    for w in range(out_w):


                        dv_x[f, h*h_stride:h*h_stride + f_h, w*w_stride:w*w_stride + f_w] += dv_y[c, h, w] * self.W[c, f]

                        dv_W[c, f] += x[f, h*h_stride:h*h_stride + f_h, w*w_stride:w*w_stride + f_w]*dv_y[c, h, w]
                        dv_b[f] += dv_y[c, h, w]
                        print(dv_x[c, h*h_stride:h*h_stride + f_h, w*w_stride:w*w_stride + f_w])

        # don't change the order of return values
        return dv_x, dv_W, dv_b


class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
            'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
            'Width doesn\'t work'

        y_shape = (
            x.shape[0],
            int((x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0]) + 1,
            int((x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1]) + 1,
        )
        y = np.empty(y_shape, dtype=np.float32)

        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                y[:, i, j] = np.max(x_padded[
                                    :,
                                    i * s[0]: i * s[0] + self.kernel_size[0],
                                    j * s[1]: j * s[1] + self.kernel_size[1]
                                    ].reshape(-1, self.kernel_size[0] * self.kernel_size[1]),
                                    axis=1
                                    )

        return y

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
                respect to the input.

                Args:
                    x (np.array): the input of the layer. A 3D array of shape (
                        in_channels, in_heights, in_weights).
                    dv_y (np.array): The derivative of the loss with respect to the
                        output. A 3D array of shape (out_channels, out_heights,
                        out_weights).

                Returns:
                    dv_x (np.array): The derivative of the loss with respect to the
                        input. It has the same shape as x.
                """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # TODO: write your implementation below
        dv_x = np.empty(x.shape, dtype=np.float32)


        out_c, out_h, out_w = dv_y.shape
        h_stride = s[0]
        w_stride = s[1]
        h_pad = p[0]
        w_pad = p[1]

        for c in range(out_c):
            for h in range(out_h):
                for w in range(out_w):
                    x_region = x_padded[c, h*h_stride:h*h_stride + self.kernel_size[0], w*w_stride:w*w_stride
                                                                                            + self.kernel_size[1]]
                    mask = (x_region == np.max(x_region))
                    dv_x[c, h*h_stride:h*h_stride + self.kernel_size[0], w*w_stride:w*w_stride + self.kernel_size[1]] \
                        = mask*dv_y[c, h, w]

        return dv_x

