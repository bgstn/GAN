import tensorflow as tf


class ModelBase:
    def __init__(self):
        self.name = 'ModelBase'

    def get_models(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class Discriminator(ModelBase):
    def __init__(self):
        super().__init__()
        self.name = "discriminator"

    def get_models(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class Generator(ModelBase):
    def __init__(self):
        super().__init__()
        self.name = "generator"

    def get_models(self):
        raise NotImplementedError

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
