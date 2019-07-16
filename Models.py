import tensorflow as tf
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, concatenate


class ModelBase:
    def __init__(self, Input_dims):
        self.name = 'ModelBase'
        self.Input_dims = Input_dims

    def get_models(self):
        pass

    def train(self):
        raise NotImplementedError


class Discriminator(ModelBase):
    def __init__(self, num_pixels, num_classes):
        super().__init__()
        self.name = "discriminator"
        self.num_pixels = num_pixels
        self.num_classes = num_classes

    def get_models(self):
        inpt1 = Input(shape=(self.num_pixels,))
        dense1 = Dense(self.num_pixels, kernel_initializer='normal',activation='relu')(inpt1)

        inpt2 = Input(shape=(self.num_pixels,))
        dense2 = Dense(self.num_pixels,kernel_initializer='normal',activation='relu')(inpt2)

        concat_layer = concatenate([dense1, dense2], axis=-1)
        logits = Dense(self.num_classes, kernel_initializer='normal',activation='softmax')(concat_layer)
        model = Model([inpt1, inpt2], logits)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    def train(self):
        raise NotImplementedError


class Generator(ModelBase):
    def __init__(self):
        super().__init__()
        self.name = "generator"
        self.hidden_size = [256, 256]

    def get_models(self, discriminator):
        inpt1 = Input(shape=self.input_dimension)
        l = inpt1
        for hid in self.hidden_size:
            l = Dense(units=hid, kernel_initializer='normal',activation='relu')(l)
        model_d = clone_model(model=discriminator)
        for layer in model_d.layers:
            layer.trainable = False

        inpt2 = Input(shape=(784,))
        out = model_d([l, inpt2])
        model = Model([inpt1, inpt2], out)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    def train(self):
        raise NotImplementedError


class GAN(ModelBase):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.name = 'GAN'
        self.discriminator = discriminator
        self.generator = generator

    def get_models(self):
        # get models
        discriminator = self.discriminator.get_models()
        generator = self.generator.get_models()

        # re-ensemble two networks for training

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
