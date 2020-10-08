"""
Run as:
    python main.py --method <met> --epochs <n_epochs>
    where met can be piven, qd or only_point (NN in the paper)
    default n_epochs = 50

    ground-truth source: https://www.kaggle.com/kmader/rsna-bone-age/discussion/145094
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import random
import argparse
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.layers as KL
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.metrics import mean_absolute_error
from keras import Model
from keras.initializers import RandomNormal, Constant
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 256
RESULTS_PATH = 'results-piven'
seed_value = 1
base_bone_dir = 'rsna-bone-age'


def get_mean_std():
    """
    @return: mean and std for normalization
    """
    train_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
    test_df = pd.read_excel(os.path.join(base_bone_dir, 'Bone age ground truth.xlsx'))
    y_train = train_df['boneage'].values.astype(np.float64)
    y_test = test_df['Ground truth bone age (months)'].values.astype(np.float64)
    y_train_test = np.concatenate([y_train, y_test])
    return y_train_test.mean(), y_train_test.std()


def load_data():
    """
    @return: data generators (train, val and test)
    """
    train_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))
    test_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-test-dataset.csv'))

    m, v = get_mean_std()
    train_df['id'] = train_df['id'].apply(lambda x: str(x) + '.png')
    test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x) + '.png')
    train_df['bone_age_z'] = (train_df['boneage'] - m) / v
    train_df['boneage_category'] = pd.cut(train_df['boneage'], 10)

    df_train, df_valid = train_test_split(train_df,
                                          test_size=0.1,
                                          random_state=seed_value,
                                          stratify=train_df['boneage_category'])

    train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    # train data generator
    train_dir = './rsna-bone-age/boneage-training-dataset/boneage-training-dataset'
    train_gen = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        directory=train_dir,
        x_col='id',
        y_col='bone_age_z',
        batch_size=50,
        shuffle=True,
        class_mode='other',
        flip_vertical=True,
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE))

    # validation data generator
    val_gen = val_data_generator.flow_from_dataframe(
        dataframe=df_valid,
        directory=train_dir,
        x_col='id',
        y_col='bone_age_z',
        batch_size=50,
        shuffle=True,
        class_mode='other',
        flip_vertical=True,
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE))

    # test data generator
    test_gen = test_data_generator.flow_from_directory(
        directory='./rsna-bone-age/boneage-test-dataset',
        shuffle=False,
        batch_size=50,
        class_mode=None,
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE))

    return train_gen, val_gen, test_gen


def build_model(method, boneage_div, lambda_in=15.0, soften=160.0, alpha=0.05):
    """
    Build the model using 'method' loss (piven, qd or only_point)

    @param method: method name
    @param boneage_div: data std
    @param soften: soften parameter for qd loss
    @param lambda_in: lambda parameter for qd loss
    @param alpha: confidence level
    @return: complied model
    """

    def mae_months(y_true, y_pred):
        if method == 'only_point':
            return mean_absolute_error(boneage_div * y_true, boneage_div * y_pred)

        y_true = y_true[:, 0]
        y_u_pred = y_pred[:, 0]
        y_l_pred = y_pred[:, 1]

        if method == 'piven':
            y_v = y_pred[:, 2]
            y_eli = y_v * y_u_pred + (1 - y_v) * y_l_pred
        if method == 'qd':
            y_eli = 0.5 * y_u_pred + 0.5 * y_l_pred

        return mean_absolute_error(boneage_div * y_true, boneage_div * y_eli)

    def mpiw(y_true, y_pred):
        y_u_pred = y_pred[:, 0]
        y_l_pred = y_pred[:, 1]
        mpiw = tf.reduce_mean(y_u_pred - y_l_pred)
        return mpiw

    def picp(y_true, y_pred):
        y_true = y_true[:, 0]
        y_u_pred = y_pred[:, 0]
        y_l_pred = y_pred[:, 1]
        K_u = tf.cast(y_u_pred > y_true, tf.float32)
        K_l = tf.cast(y_l_pred < y_true, tf.float32)
        picp = tf.reduce_mean(K_l * K_u)
        return picp

    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    input_layer = KL.Input(input_shape)
    base_pretrained_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = False
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(input_layer)
    bn_features = KL.BatchNormalization()(pt_features)

    attn_layer = KL.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
    attn_layer = KL.Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = KL.LocallyConnected2D(1,
                                       kernel_size=(1, 1),
                                       padding='valid',
                                       activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = KL.Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                      activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = KL.multiply([attn_layer, bn_features])
    gap_features = KL.GlobalAveragePooling2D()(mask_features)
    gap_mask = KL.GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = KL.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = KL.Dropout(0.5)(gap)
    x = KL.Dropout(0.25)(KL.Dense(1024, activation='elu')(gap_dr))

    point = KL.Dense(1, activation='linear')(x)
    pi = KL.Dense(2, activation='linear', kernel_initializer=RandomNormal(stddev=0.1),
                  bias_initializer=Constant(value=[2.0, -2.0]), name='pi')(x)

    v = KL.Dense(1, activation='sigmoid', name='v', bias_initializer=Constant(value=[0.]))(x)
    v_pi = KL.Concatenate(name='v_pi_concat')([pi, v])

    if method == 'piven':
        out = v_pi
        metrics = [picp, mpiw, mae_months]
        loss = piven_loss(True, lambda_in, soften, alpha)
    elif method == 'qd':
        out = pi
        metrics = [picp, mpiw, mae_months]
        loss = piven_loss(False, lambda_in, soften, alpha)
    elif method == 'only_point':
        out = point
        metrics = [mae_months]
        loss = 'mse'

    bone_age_model = Model(inputs=[input_layer], outputs=[out])

    # compile model
    opt = Adam(lr=0.001)
    bone_age_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # model summary
    bone_age_model.summary()

    return bone_age_model


def piven_loss(eli, lambda_in=15.0, soften=160.0, alpha=0.05):
    # define loss fn
    def piven_loss(y_true, y_pred):
        y_U = y_pred[:, 0]
        y_L = y_pred[:, 1]
        y_T = y_true[:, 0]

        if eli:
            y_v = y_pred[:, 2]

        N_ = tf.cast(tf.size(y_T), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)

        # soft uses sigmoid
        k_soft = tf.multiply(tf.sigmoid((y_U - y_T) * soften), tf.sigmoid((y_T - y_L) * soften))

        # hard uses sign step function
        k_hard = tf.multiply(tf.maximum(0., tf.sign(y_U - y_T)), tf.maximum(0., tf.sign(y_T - y_L)))

        MPIW_capt = tf.divide(tf.reduce_sum(tf.abs(y_U - y_L) * k_hard),
                              tf.reduce_sum(k_hard) + 0.001)

        PICP_soft = tf.reduce_mean(k_soft)

        qd_rhs_soft = lambda_ * tf.sqrt(N_) * tf.square(tf.maximum(0., (1. - alpha_) - PICP_soft))
        piven_loss_ = MPIW_capt + qd_rhs_soft  # final qd loss form

        if eli:
            y_eli = y_v * y_U + (1 - y_v) * y_L
            y_eli = tf.reshape(y_eli, (-1, 1))
            piven_loss_ += tf.losses.mean_squared_error(y_true, y_eli)

        return piven_loss_

    return piven_loss


def train(model, train_gen, val_gen, epochs=100):
    weight_path = RESULTS_PATH + "/bone_age_weights_{epoch:02d}.hdf5"

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, verbose=1, mode='auto',
                                    epsilon=0.0001, cooldown=5, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1, mode='min')

    #     callbacks_list = [checkpoint, early_stopping, red_lr_plat]
    callbacks_list = [checkpoint, red_lr_plat]

    history = model.fit_generator(train_gen,
                                  verbose=1,
                                  validation_data=val_gen,
                                  epochs=epochs,
                                  callbacks=callbacks_list)

    return history


def predict(method, model, test_gen):
    y_pred = model.predict_generator(test_gen, steps=4)
    if method == 'only_point':
        return y_pred

    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    if method == 'piven':
        y_v = y_pred[:, 2]
        y_eli = y_v * y_u_pred + (1 - y_v) * y_l_pred
        return y_l_pred, y_u_pred, y_eli

    # qd
    return y_l_pred, y_u_pred, 0.5 * (y_l_pred + y_u_pred)


def evaluate(method, model, test_gen):
    from sklearn.metrics import mean_absolute_error
    y_pred = predict(method, model, test_gen)

    m, v = get_mean_std()
    test_df = pd.read_excel(os.path.join('rsna-bone-age', 'Bone age ground truth.xlsx'))
    y_true = test_df['Ground truth bone age (months)'].values.astype(np.float64)
    y_true = (y_true - m) / v

    if method == 'only_point':
        # mae months
        mad = mean_absolute_error(v * y_true, v * y_pred)
        return -1, -1, mad

    y_l_pred, y_u_pred, y_point_pred = y_pred

    # picp
    K_u = y_u_pred > y_true
    K_l = y_l_pred < y_true
    picp = np.mean(K_l * K_u)

    # mpiw
    mpiw = np.mean(y_u_pred - y_l_pred)

    # mae months
    mad = mean_absolute_error(v * y_true, v * y_point_pred)
    return picp, mpiw, mad


def fix_seeds():
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.set_random_seed(seed_value)


def main():
    fix_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, help='qd, piven, only_point', required=True)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    global RESULTS_PATH

    if args.method == 'piven':
        RESULTS_PATH = './results-piven'
    elif args.method == 'qd':
        RESULTS_PATH = './results-qd'
    elif args.method == 'only_point':
        RESULTS_PATH = './results-only_point'
    else:
        raise ValueError(f"Unknown method: {args.method}")

    train_gen, val_gen, test_gen = load_data()

    m, v = get_mean_std()
    model = build_model(method=args.method, boneage_div=v)

    history = train(model, train_gen, val_gen, epochs=args.epochs)

    with open(f'{RESULTS_PATH}/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    best_weights_file = max(glob.glob(f'{RESULTS_PATH}/*.hdf5'), key=os.path.getmtime)  # last file the written
    model.load_weights(best_weights_file)

    picp_m, mpiw_m, mad_m = evaluate(args.method, model, test_gen)
    ans_str = f"{args.method}: \npicp: {picp_m}\nmpiw: {mpiw_m}\nmad: {mad_m}"
    with open(f"{RESULTS_PATH}/eval.txt", "w") as f:
        f.write(ans_str)
    print(ans_str)


if __name__ == '__main__':
    main()
