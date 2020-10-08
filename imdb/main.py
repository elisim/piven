"""
Run as:
    python main.py --input imdb.npz --db imdb --method <met>
    where met is piven, qd or only_point (NN in the paper)
"""

import pandas as pd
import argparse
import os
import glob
import train_callbacks

from sklearn.model_selection import KFold, train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from model import *
from utils import mk_dir, load_data_npz
from generators import *


db_name = 'imdb'
seed = 2 ** 10
np.random.seed(seed)
logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input database npz file")
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--method", type=str, required=True,
                        help="only_point, qd, eli, sqr")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")

    args = parser.parse_args()
    return args


def get_model(method, image_size, v):
    optMethod = Adam()
    N_densenet = 3
    depth_densenet = 3 * N_densenet + 4

    if method == 'piven':
        model = PI_Model(image_size, depth_densenet, method='piven')()
        save_name = 'densenet_eli_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=optMethod, loss=piven_loss(eli=True), metrics=[picp, mpiw, mae(method, v)])

    elif method == 'qd':
        model = PI_Model(image_size, depth_densenet, method='qd')()
        save_name = 'densenet_qd_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=optMethod, loss=piven_loss(eli=False), metrics=[picp, mpiw, mae(method, v)])

    elif method == 'only_point':
        model = DenseNet_reg(image_size, depth_densenet)()
        save_name = 'densenet_reg_%d_%d' % (depth_densenet, image_size)
        model.compile(optimizer=optMethod, loss=["mae"], metrics={'pred_a': 'mae'})

    else:
        raise ValueError(f"Invalid method name: {method}")

    logging.debug("Model summary...")
    model.count_params()
    model.summary()

    db_name = 'imdb'
    logging.debug("Saving model...")
    mk_dir(db_name + "_models")
    mk_dir(db_name + "_models/" + save_name)
    mk_dir(db_name + "_checkpoints")
    mk_dir(db_name + f"_checkpoints/{method}")

    with open(os.path.join(db_name + "_models/" + save_name, save_name + '.json'), "w") as f:
        f.write(model.to_json())

    return model, save_name


def fit_model(model, X_train, y_train, X_val, y_val, nb_epochs, batch_size, save_name, run, method):
    start_decay_epoch = [30, 60]
    decaylearningrate = train_callbacks.DecayLearningRate(start_decay_epoch)

    mk_dir(db_name + f"_checkpoints/{method}/{run}")

    callbacks = [ModelCheckpoint(db_name + f"_checkpoints/{method}/{run}" + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                 ]

    hist = model.fit_generator(generator=data_generator_reg(X=X_train, Y=y_train, batch_size=batch_size),
                               steps_per_epoch=X_train.shape[0] // batch_size,
                               validation_data=(X_val, [y_val]),
                               epochs=nb_epochs, verbose=1,
                               callbacks=callbacks)

    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name + "_models/" + save_name, save_name + '.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name + "_models/" + save_name, 'history_' + save_name + '.h5'),
                                      "history")


def main():
    args = get_args()
    input_path = args.input
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split

    logging.debug("Loading data...")
    image, gender, age, image_size = load_data_npz(input_path)

    x_data = image
    y_data_a = age
    m, v = np.mean(y_data_a), np.std(y_data_a)
    if args.method == 'qd' or args.method == 'piven':
        y_data_a = (y_data_a - m) / v

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    run = 0

    for train, test in kfold.split(x_data, y_data_a):
        logging.debug(f"Running training {run}...")

        # create model
        model, save_name = get_model(args.method, image_size, v)

        # val split
        X_train, X_val, y_train, y_val = train_test_split(x_data[train], y_data_a[train],
                                                          test_size=validation_split,
                                                          random_state=seed)

        # Fit the model
        fit_model(model, X_train, y_train, X_val, y_val, nb_epochs, batch_size, save_name, run, args.method)

        # last file that written
        best_weights_file = max(glob.glob(f"{db_name}_checkpoints/{args.method}/{run}/*.hdf5"), key=os.path.getmtime)
        model.load_weights(best_weights_file)

        # evaluate the model
        scores = model.evaluate(x_data[test], y_data_a[test], verbose=0)

        results.append(scores)
        run += 1

    results_to_csv(results, 5, os.path.dirname(input_path), args.method)


def results_to_csv(results, n_runs, path, method):
    avg = np.mean(results, axis=0)
    std_dev = np.std(results, axis=0)
    std_err = std_dev / np.sqrt(n_runs)

    if len(avg) > 2:
        index = ['loss', 'picp', 'mpiw', 'mae']
    else:
        index = ['loss', 'mae']

    data = {'avg': np.round(avg, 3), 'std_err': np.round(std_err, 3), 'std_dev': np.round(std_dev, 3)}
    data_df = pd.DataFrame(data, index=index)
    data_df.to_csv(os.path.join(path, f"results_{method}_{n_runs}_runs.csv"))


if __name__ == '__main__':
    main()
