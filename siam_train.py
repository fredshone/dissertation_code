"""Example Association Model experiment"""

from keras import Model
from keras import layers
from keras import optimizers
from keras import Input
from keras import callbacks
from keras.applications import ResNet50
import generator as gen
import argparse


def build_model(input_shape=(197, 197, 3)):
    """
    Builds model for training.
    :param input_shape: Tuple of input image size (height, width, channels) (default (197,197,3)).
    :return: Returns keras model
    """
    print('Loading base CNN model...')

    cnn_base = ResNet50(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape)

    cnn_base.trainable = False

    print('Building model...')

    sample1_input = Input(shape=input_shape)
    sample2_input = Input(shape=input_shape)
    sample1_cnn = cnn_base(sample1_input)
    sample2_cnn = cnn_base(sample2_input)
    sample1_cnn_flatten = layers.Flatten()(sample1_cnn)
    sample2_cnn_flatten = layers.Flatten()(sample2_cnn)
    cnn_top = layers.Dense(1024, activation="relu")
    sample1_cnn_top = cnn_top(sample1_cnn_flatten)
    sample2_cnn_top = cnn_top(sample2_cnn_flatten)
    diff_features = layers.Subtract()([sample1_cnn_top, sample2_cnn_top])
    diff_features_norm = layers.BatchNormalization()(diff_features)
    top1 = layers.Dense(512, activation='relu')(diff_features_norm)
    top2 = layers.Dense(256, activation='relu')(top1)
    answer = layers.Dense(1, activation='sigmoid')(top2)

    model = Model([sample1_input, sample2_input], answer)

    print(model.summary())

    return model


def train(batch_size=256, input_shape=(197, 197, 3), epochs=100, steps_epoch=10, val_steps=5, verbose=False, debug=False, load_weights=None):
    """
    Function to train given model and save training checkpoints.
    :param batch_size: Integer, batch size (default 256)
    :param input_shape: Tuple of input image size (height, width, channels) (default (197,197,3)).
    :param epochs: Integer, maximum epochs (default 100)
    :param steps_epoch: Integer, steps per epoch (default 10).
    :param val_steps: Integer, batches for validation per epoch (default 5).
    :param verbose: Boolean (default False).
    :param debug: Boolean (default False).
    :param load_weights: String, optional, path to existing weights if used (default None).
    :return: None
    """
    siam_model = build_model(input_shape)
    if load_weights:
        if verbose:
            print('Loading weights from {}'.format(load_weights))
        siam_model.load_weights(load_weights)
        if verbose:
            print('...done')

    optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    siam_model.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['acc'])
    training_gen = gen.generator(use='train', image_size=input_shape, batch_size=batch_size, verbose=debug)
    validation_gen = gen.generator(use='validate', image_size=input_shape, batch_size=batch_size, verbose=debug)
    filepath = "weights.{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                           monitor='val_acc',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='max',
                                           period=1)
    early = callbacks.EarlyStopping(monitor='val_acc',
                                    min_delta=0,
                                    patience=20,
                                    verbose=1,
                                    mode='max',
                                    baseline=None)
    callbacks_list = [checkpoint, early]
    history = siam_model.fit_generator(generator=training_gen,
                                       steps_per_epoch=steps_epoch,
                                       epochs=epochs,
                                       validation_data=validation_gen,
                                       validation_steps=val_steps,
                                       callbacks=callbacks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--input_shape', default=(197, 197, 3))
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--steps_epoch', default=10)
    parser.add_argument('--val_steps', default=2)
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-D', '--debug_generator', action='store_true', help='Print generator actions')
    parser.add_argument('--load_weights', type=str, help='File location to Load existing weights from')
    args = parser.parse_args()

    print('Training with batchsize:{}, epochs:{}, steps per epoch:{}'.
          format(args.batch_size, args.epochs, args.steps_epoch))

    train(batch_size=args.batch_size,
        input_shape=args.input_shape,
        epochs=args.epochs,
        steps_epoch=args.steps_epoch,
        val_steps=args.val_steps,
        verbose=args.verbose,
        debug=args.debug_generator,
        load_weights=args.load_weights)
