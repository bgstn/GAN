from keras.models import Model, clone_model
from keras.layers import Input, Dense, concatenate
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from tqdm import tqdm


class ModelBase:
    def __init__(self, input_dims):
        self.name = 'ModelBase'
        self.input_dims = input_dims

    def get_models(self):
        pass

    def train(self):
        pass


class Discriminator(ModelBase):
    def __init__(self, input_dims, num_classes):
        super().__init__(input_dims)
        self.name = "discriminator"
        self.num_classes = num_classes
        self.model = self.get_models()

    def get_models(self):
        inpt1 = Input(shape=self.input_dims)
        dense1 = Dense(1024, kernel_initializer='normal', activation='relu')(inpt1)

        inpt2 = Input(shape=self.input_dims)
        dense2 = Dense(1024, kernel_initializer='normal',activation='relu')(inpt2)

        concat_layer = concatenate([dense1, dense2], axis=-1)
        logits = Dense(self.num_classes, kernel_initializer='normal',activation='softmax')(concat_layer)
        model = Model([inpt1, inpt2], logits)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        raise NotImplementedError


class Generator(ModelBase):
    def __init__(self, input_dims):
        super().__init__(input_dims)
        self.name = "generator"
        self.hidden_size = [256, 256]
        self.output_size = 784
        self.model = self.get_models()

    def get_models(self):
        inpt = Input(shape=(self.input_dims,))
        l = inpt
        for hid in self.hidden_size:
            l = Dense(units=hid, kernel_initializer='normal', activation='relu')(l)
        l = Dense(units=self.output_size, kernel_initializer="normal", activation=None)(l)
        model_g = Model(inpt, l)
        return model_g

    def train(self):
        raise NotImplementedError


class GAN(ModelBase):
    def __init__(self, input_dims, num_classes):
        super().__init__(input_dims)
        self.name = 'GAN'
        self.num_classes = num_classes
        self.D = Discriminator(self.input_dims['D'], self.num_classes)
        self.G = Generator(self.input_dims['G'])
        self.model = self.get_model()

    def get_model(self):
        # re-ensemble two networks for training
        model_g = self.G.model
        model_d = clone_model(model=self.D.model)
        model_d.name = 'D_copy'
        for layer in model_d.layers:
            layer.trainable = False

        inpt1 = model_g.input
        inpt2 = Input(shape=self.input_dims['D'])

        gen_out = model_g.output
        out = model_d([gen_out, inpt2])
        model_gan = Model([inpt1, inpt2], out)
        model_gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model_gan

    def train(self, X_train):
        model_d = self.D.model
        model_g = self.G.model
        model_gan = self.model

        for e in range(100):
            print("\n\n"
                  "#################################################\n"
                  "                                                 \n"
                  "                  Epoch: {}                      \n" 
                  "                                                 \n"
                  "#################################################\n\n".format(e))
            Noise_x = np.random.normal(loc=0, scale=1, size=(X_train.shape[0], self.input_dims['G']))
            fake_image = model_g.predict(Noise_x)
            X_train_sync = np.concatenate([fake_image, X_train], axis=0)
            y_train_sync = np_utils.to_categorical(np.array([0] * fake_image.shape[0] + [1] * X_train.shape[0]))

            idx = np.random.choice(np.arange(X_train.shape[0]), replace=False, size=X_train.shape[0])
            X_train_ori = np.concatenate([X_train, X_train[idx]], axis=0)

            idx = np.random.choice(np.arange(X_train_sync.shape[0]), replace=False, size=X_train_sync.shape[0])
            X_train_sync = X_train_sync[idx, :]
            X_train_ori = X_train_ori[idx, :]

            y_train_sync = y_train_sync[idx, :]

            bar = tqdm(range(10), ascii=True, desc="Model_D")
            for _ in bar:
                pbar = tqdm(range(0, X_train_sync.shape[0], 64), ascii=True)
                for i in pbar:
                    loss, acc = model_d.train_on_batch(x=[X_train_sync[i:i + 64, :], X_train_ori[i:i + 64, :]],
                                                       y=y_train_sync[i:i + 64])
                    pbar.set_description("  Batch")
                    pbar.set_postfix({"trainloss": round(loss, 2), "acc": round(acc, 2)})
                bar.set_postfix({'train_loss': round(loss, 2), "acc": round(acc, 2)})

            model_gan.get_layer(name='D_copy').set_weights(model_d.get_weights())

            pbar = tqdm(np.random.choice(range(0, Noise_x.shape[0], 64), replace=False, size=30),
                        ascii=True, desc="Model_G")
            for i in pbar:
                loss, acc = model_gan.train_on_batch(x=[Noise_x[i:i + 64, :], X_train[i:i + 64, :]],
                                                     y=np.array([[0, 1]] * Noise_x[i:i + 64, :].shape[0]))
                pbar.set_postfix({"trainloss": round(loss, 2), "acc": round(acc, 2)})
                # print(acc, end=", ")
            # print()


if __name__ == '__main__':
    # dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    X_test = X_test.reshape(X_test.shape[0], -1) / 255

    # X_train = X_train[y_train == 0]
    # X_test = X_test[y_test == 0]
    # y_train = y_train[y_train == 0]
    # y_test = y_test[y_test == 0]

    # params
    num_pixels = X_train.shape[1]
    num_classes = 2

    # build model
    gan = GAN(input_dims={'D': (784,), 'G': 100}, num_classes=2)
    gan.model.summary()

    # train
    gan.train(X_train)

