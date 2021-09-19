import sys
import numpy as np
import tensorflow as tf
from utils import parse_args, fit_fraction


class DatasetLoader:
    def __init__(self, tf_records_path, batch_size=32, epochs=1):
        self.records_dict = {'data': tf.io.FixedLenFeature([6 * 5], tf.float32)}

        dataset = tf.data.TFRecordDataset(tf_records_path).shuffle(batch_size * 10).map(self.__map_fuc)
        dataset = dataset.batch(batch_size).repeat(epochs).prefetch(batch_size * 10)
        self.dataset = iter(dataset)

    def __map_fuc(self, tf_records):
        records = tf.io.parse_single_example(tf_records, self.records_dict)
        data = records['data']
        data = tf.reshape(data, (6, 5))
        particle_identity = tf.cast(data[:, 0], tf.int32) + 2
        particle_momentum = tf.cast(data[:, 1:], tf.float32)
        mask = tf.cast(tf.reduce_sum(tf.abs(particle_momentum), axis=-1) != 0, tf.float32)
        particle_momentum = particle_momentum / tf.reduce_max(tf.abs(particle_momentum), 0, True)
        return particle_identity, particle_momentum, mask

    def get_batch(self):
        return next(self.dataset)


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        seq_length, model_dim = input_shape[1:]
        position_encodings = np.zeros((seq_length, model_dim))
        for pos in range(seq_length):
            for i in range(model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i - i % 2) / model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2])
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2])
        self.position = tf.cast(position_encodings, tf.float32)

    def call(self, inputs, **kwargs):
        return inputs + self.position


class LayerNormal(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        super(LayerNormal, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight('gamma', (input_shape[-1],), initializer=tf.initializers.get('ones'))
        self.beta = self.add_weight('beta', (input_shape[-1],), initializer=tf.initializers.get('zeros'))
        super(LayerNormal, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, var = tf.nn.moments(inputs, -1, keepdims=True)
        normal = (inputs - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * normal + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, head_nums, head_size, out_dim=None, key_size=None, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_nums = head_nums
        self.head_size = head_size
        self.out_dim = out_dim or head_nums * head_size
        self.key_size = key_size or head_size
        self.scale = tf.sqrt(tf.cast(self.key_size, tf.float32))

    def build(self, input_shape):
        self.query_weight = self.add_weight('query_weight', (input_shape[0][-1], self.key_size * self.head_nums),
                                            initializer=tf.initializers.get('glorot_uniform'))
        self.query_bias = self.add_weight('query_bias', (self.key_size * self.head_nums,), initializer=tf.initializers.get('zeros'))
        self.key_weight = self.add_weight('key_weight', (input_shape[1][-1], self.key_size * self.head_nums),
                                          initializer=tf.initializers.get('glorot_uniform'))
        self.key_bias = self.add_weight('key_bias', (self.key_size * self.head_nums,), initializer=tf.initializers.get('zeros'))
        self.value_weight = self.add_weight('value_weight', (input_shape[2][-1], self.head_size * self.head_nums),
                                            initializer=tf.initializers.get('glorot_uniform'))
        self.value_bias = self.add_weight('value_bias', (self.head_size * self.head_nums,), initializer=tf.initializers.get('zeros'))
        self.concat_weight = self.add_weight('concat_weight', (self.head_size * self.head_nums, self.out_dim),
                                             initializer=tf.initializers.get('glorot_uniform'))
        self.concat_bias = self.add_weight('concat_bias', (self.out_dim,), initializer=tf.initializers.get('zeros'))
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, key, value = inputs[:3]
        mask = inputs[3:]
        query = tf.einsum("bni,ij->bnj", query, self.query_weight) + self.query_bias
        key = tf.einsum("bni,ij->bnj", key, self.key_weight) + self.key_bias
        value = tf.einsum("bni,ij->bnj", value, self.value_weight) + self.value_bias

        q = tf.reshape(query, (-1, query.shape[1], self.head_nums, self.key_size))
        k = tf.reshape(key, (-1, key.shape[1], self.head_nums, self.key_size))
        v = tf.reshape(value, (-1, value.shape[1], self.head_nums, self.head_size))

        o = self.apply_attention_to([q, k, v], mask)
        o = tf.reshape(o, (-1, o.shape[1], self.head_nums * self.head_size))
        o = tf.einsum("bni,ij->bnj", o, self.concat_weight) + self.concat_bias
        return o

    def apply_attention_to(self, inputs, mask):
        q, k, v = inputs
        mask_nums = len(mask)
        if mask_nums == 0:
            pad_mask, word_mask = None, None
        elif mask_nums == 1:
            pad_mask, word_mask = mask[0], None
        else:
            pad_mask, word_mask = mask

        a = tf.einsum("bnhd,bmhd->bhnm", q, k) / self.scale
        if pad_mask is not None:
            a = a - (1.0 - pad_mask[:, tf.newaxis, tf.newaxis, :]) * 1e12
        if word_mask is not None:
            a = a - (1.0 - word_mask[:, tf.newaxis]) * 1e12
        a = tf.nn.softmax(a, -1)
        o = tf.einsum("bhnm,bmhd->bnhd", a, v)
        return o

    def compute_output_shape(self, input_shape):
        return [input_shape[0][0], input_shape[0][1], self.out_dim]


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, unit, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.unit = unit

    def build(self, input_shape):
        self.inner_weight = self.add_weight('inner_weight', (input_shape[-1], self.unit), initializer=tf.initializers.get('glorot_uniform'))
        self.inner_bias = self.add_weight('inner_bias', (self.unit,), initializer=tf.initializers.get('zeros'))
        self.outer_weight = self.add_weight('outer_weight', (self.unit, input_shape[-1]), initializer=tf.initializers.get('glorot_uniform'))
        self.outer_bias = self.add_weight('outer_bias', (input_shape[-1],), initializer=tf.initializers.get('zeros'))
        super(FeedForward, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner = tf.nn.relu(tf.einsum("bni,ij->bnj", inputs, self.inner_weight) + self.inner_bias)
        outer = tf.einsum("bni,ij->bnj", inner, self.outer_weight) + self.outer_bias
        return outer


class Normalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Normalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), -1, True) + 1e-6)


class LinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, scale=2, **kwargs):
        super(LinearInterpolation, self).__init__(**kwargs)
        self.scale = scale

    def build(self, input_shape):
        rate = (input_shape[-1] - 1) / (int(input_shape[-1] * self.scale) - 1)
        corr = tf.cast(tf.range(0, int(input_shape[-1] * self.scale)), tf.float32)
        center = tf.reshape(rate * corr, [1, 1, int(input_shape[-1] * self.scale)])
        self.center = tf.tile(center, [1, input_shape[1], 1])
        self.low = tf.math.floor(self.center)
        self.top = tf.math.ceil(self.center)
        super(LinearInterpolation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        indices = tf.stack(tf.meshgrid(tf.range(0, shape[0]), tf.range(0, shape[1]), indexing='ij'), -1)[..., tf.newaxis, :]
        indices = tf.cast(tf.tile(indices, [1, 1, self.center.shape[-1], 1]), tf.float32)
        low_indices = tf.concat([indices, tf.tile(self.low[..., tf.newaxis], [tf.shape(inputs)[0], 1, 1, 1])], -1)
        top_indices = tf.concat([indices, tf.tile(self.top[..., tf.newaxis], [tf.shape(inputs)[0], 1, 1, 1])], -1)
        low_value = tf.gather_nd(inputs, tf.cast(low_indices, tf.int32))
        top_value = tf.gather_nd(inputs, tf.cast(top_indices, tf.int32))
        value = tf.where(self.low == self.top, low_value,
                         ((self.top - self.center) * low_value + (self.center - self.low) * top_value) / (self.top - self.low + 1e-3))
        return value


class StyleBlock2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, demod=True, upsample=False, **kwargs):
        super(StyleBlock2, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.demod = demod
        self.upsample = upsample

    def build(self, input_shape):
        latent_shape, content_shape = input_shape
        self.affine_weight = self.add_weight('affine_weight', (latent_shape[-1], content_shape[1]), initializer=tf.initializers.get('glorot_uniform'))
        self.affine_bias = self.add_weight('affine_bias', (1, content_shape[1]), initializer=tf.initializers.get('zeros'))
        self.kernel_weight = self.add_weight('kernel_weight', (self.kernel_size, content_shape[1], self.filters), initializer=tf.initializers.get('he_normal'))
        self.kernel_bias = self.add_weight('kernel_bias', (1, self.filters, 1), initializer=tf.initializers.get('zeros'))
        if self.upsample:
            self.linear = LinearInterpolation()
        super(StyleBlock2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        latent, content = inputs
        affine = tf.einsum("bi,ij->bj", latent, self.affine_weight) + self.affine_bias
        kernel_weight = self.kernel_weight[tf.newaxis] * affine[:, tf.newaxis, :, tf.newaxis]
        if self.demod:
            kernel_weight = kernel_weight / tf.sqrt(tf.reduce_sum(tf.square(kernel_weight), (1, 2), True) + 1e-8)
        weight = tf.reshape(tf.transpose(kernel_weight, [1, 2, 0, 3]), [kernel_weight.shape[1], kernel_weight.shape[2], -1])
        if self.upsample:
            content = self.linear(content)
        content = tf.reshape(content, [1, -1, content.shape[-1]])
        feature = tf.nn.conv1d(content, weight, [1, 1, 1], 'SAME', 'NCW')
        feature = tf.reshape(feature, [-1, self.filters, feature.shape[-1]]) + self.kernel_bias
        return tf.nn.leaky_relu(feature, 0.2)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.linear1 = LinearInterpolation(0.5)
        self.first_conv = tf.keras.layers.Conv1D(self.filters, 3, 1, 'same', 'channels_first', activation=tf.nn.leaky_relu, kernel_initializer='he_normal')
        self.second_conv = tf.keras.layers.Conv1D(self.filters, 3, 1, 'same', 'channels_first', activation=tf.nn.leaky_relu, kernel_initializer='he_normal')
        self.linear2 = LinearInterpolation(0.5)
        self.short_conv = tf.keras.layers.Conv1D(self.filters, 1, 1, 'valid', 'channels_first', activation=tf.nn.leaky_relu, kernel_initializer='he_normal')
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        short = self.linear2(inputs)
        short = self.short_conv(short)
        main = self.first_conv(inputs)
        main = self.linear1(main)
        main = self.second_conv(main)
        return (short + main) / tf.sqrt(2.0)


class MergeCondition(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MergeCondition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.transfer_weight = self.add_weight('transfer_weight', (input_shape[0][-1], input_shape[1][-1]), tf.float32, tf.initializers.get('he_normal'))
        self.transfer_bias = self.add_weight('transfer_bias', (1, input_shape[1][-1]), tf.float32, tf.initializers.get('zeros'))
        super(MergeCondition, self).build(input_shape)

    def call(self, inputs, **kwargs):
        condition, feature = inputs
        transfer = tf.einsum("bi,ij->bj", condition, self.transfer_weight) + self.transfer_bias
        merge = tf.nn.leaky_relu(tf.einsum("bni,bi->bn", feature, transfer))
        return merge


class ClassifyModel:
    def __init__(self, particle_nums, embedding_size=512, head_nums=8, head_size=64, hide_dim=1024):
        identity_inputs = tf.keras.layers.Input((particle_nums,))
        momentum_inputs = tf.keras.layers.Input((particle_nums, 4))
        mask_inputs = tf.keras.layers.Input((particle_nums,))

        def get_mapping(inputs, layer_nums=4):
            x = inputs
            for _ in range(layer_nums):
                x = tf.keras.layers.Dense(embedding_size, 'relu', kernel_initializer='he_normal')(x)
                x = LayerNormal(1e-6)(x)
            return x

        def get_encoder(inputs, pad_mask, dropout_rate=0.1):
            multi = MultiHeadAttention(head_nums, head_size)([inputs, inputs, inputs, pad_mask])
            multi = tf.keras.layers.Dropout(dropout_rate)(multi)
            add = tf.keras.layers.Add()([multi, inputs])
            normal = LayerNormal(1e-6)(add)
            feed = FeedForward(hide_dim)(normal)
            feed = tf.keras.layers.Dropout(dropout_rate)(feed)
            add = tf.keras.layers.Add()([feed, normal])
            normal = LayerNormal(1e-6)(add)
            return normal

        embedding = tf.keras.layers.Embedding(10, embedding_size)(identity_inputs)
        mapping = get_mapping(momentum_inputs)
        merge = tf.keras.layers.Add()([embedding, mapping])
        position = PositionEmbedding()(merge)
        encoder = get_encoder(position, mask_inputs)
        encoder = get_encoder(encoder, mask_inputs)
        encoder = get_encoder(encoder, mask_inputs)
        encoder = get_encoder(encoder, mask_inputs)
        encoder = get_encoder(encoder, mask_inputs)
        encoder = get_encoder(encoder, mask_inputs)
        global_pool = tf.keras.layers.GlobalAveragePooling1D()(encoder)
        feature = tf.keras.layers.Activation(tf.nn.tanh)(global_pool)
        self.classify = tf.keras.Model(inputs=[identity_inputs, momentum_inputs, mask_inputs], outputs=feature)

    def get_model(self, weight_path=None):
        if weight_path is not None:
            self.load_model(weight_path)
        return self.classify

    def load_model(self, path):
        self.classify.load_weights(path + 'classify.h5', True, True)


class GenerateModel:
    def __init__(self, noise_dim, particle_nums=6, embedding_size=64, head_nums=4, head_size=16, hide_dim=128, condition_weight_dir_path=None):
        self.noise_dim = noise_dim
        self.condition = ClassifyModel(particle_nums, embedding_size, head_nums, head_size, hide_dim).get_model(condition_weight_dir_path)

        condition_inputs = tf.keras.layers.Input((embedding_size,))

        def get_mapping(inputs, unit, activate, layer_nums=8):
            x = Normalization()(inputs)
            for i in range(layer_nums):
                x = tf.keras.layers.Dense(unit, activate, kernel_initializer='he_normal')(x)
            return x

        noise_inputs = tf.keras.layers.Input((64, noise_dim))
        condition = get_mapping(condition_inputs, embedding_size, tf.nn.relu, 8)
        content = StyleBlock2(64, 3)([condition, noise_inputs])
        content = StyleBlock2(64, 3)([condition, content])
        theta = StyleBlock2(2, 1, False)([condition, content])
        content = StyleBlock2(64, 3)([condition, content])
        content = StyleBlock2(64, 3, upsample=True)([condition, content])
        add = LinearInterpolation()(theta)
        theta = StyleBlock2(2, 1, False)([condition, content])
        theta = tf.keras.layers.Add()([add, theta])
        content = StyleBlock2(32, 3)([condition, content])
        content = StyleBlock2(32, 3, upsample=True)([condition, content])
        add = LinearInterpolation()(theta)
        theta = StyleBlock2(2, 1, False)([condition, content])
        theta = tf.keras.layers.Add()([add, theta])
        content = StyleBlock2(16, 3)([condition, content])
        content = StyleBlock2(16, 3, upsample=True)([condition, content])
        add = LinearInterpolation()(theta)
        theta = StyleBlock2(2, 1, False)([condition, content])
        theta = tf.keras.layers.Add()([add, theta])

        self.generator = tf.keras.Model(inputs=[condition_inputs, noise_inputs], outputs=theta)

    def get_model(self, path=None):
        if path is not None:
            self.load_model(path)
        return self.condition, self.generator

    def load_model(self, path):
        self.condition.load_weights(path + 'classify.h5', True, True)
        self.generator.set_weights(np.load(path + 'generator.npy', allow_pickle=True))


def inference(args):
    cos_theta = []
    dataset = DatasetLoader('./dataset/' + args.dataset + '.tfrecords', 256, 1)
    condition, generator = GenerateModel(64).get_model('./weights/' + args.model_name + '/')
    while True:
        try:
            identity, momentum, mask = dataset.get_batch()
            cond = condition([identity, momentum, mask], training=False)
            noise = tf.random.normal((tf.shape(mask)[0], 64, 64))
            cos_theta.append(tf.reduce_mean(generator([cond, noise]), -1))
        except StopIteration:
            cos_theta = np.concatenate(cos_theta, 0)
            break
    theta_bins = np.histogramdd(cos_theta, 10, ((-1, 1), (-1, 1)), density=True)[0] * (2 / 10) ** 2
    fit_fraction(theta_bins, args.model_name, str(args.energy_level) + 'tev')


if __name__ == '__main__':
    inference(parse_args())
