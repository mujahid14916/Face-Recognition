from keras.callbacks import Callback
import numpy as np


class SaveModel(Callback):

    def set_parameters(self, model, period):
        self.e_model = model
        self.period = period

    def set_test_images(self, a, b, c):
        self.a = np.expand_dims(a, axis=0)
        self.b = np.expand_dims(b, axis=0)
        self.c = np.expand_dims(c, axis=0)

    # def on_train_begin(self, logs=None):
    #     pass
    #
    # def on_train_end(self, logs=None):
    #     pass
    #
    # def on_batch_begin(self, batch, logs=None):
    #     pass
    #
    # def on_batch_end(self, batch, logs=None):
    #     pass
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     pass

    def on_epoch_end(self, epoch, logs=None):
        ea = self.e_model.predict(self.a)
        eb = self.e_model.predict(self.b)
        ec = self.e_model.predict(self.c)
        print(np.linalg.norm(ea - eb), np.linalg.norm(ea - ec))

        if (epoch+1) % self.period == 0:
            print('Saving Model at epoch {}'.format(epoch+1))
            self.e_model.save('siamese_model/siamese_'+str(epoch+1)+'_'+str(logs.get('loss'))+'.h5', include_optimizer=False)

