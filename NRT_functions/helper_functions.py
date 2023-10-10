# Helper functions for neural representation theory
import os
from datetime import datetime 
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import pickle

# A function to setup a save file
def setup_save_file(parameters):
    today = datetime.strftime(datetime.now(), '%y%m%d')
    now = datetime.strftime(datetime.now(), '%H%M%S')
    # Make sure folder is there
    if not os.path.isdir(f"./data/"):
        os.mkdir(f"./data/")
    if not os.path.isdir(f"data/{today}/"):
        os.mkdir(f"data/{today}/")
    # Now make a folder in there for this run
    savepath = f"data/{today}/{now}/"
    if not os.path.isdir(f"data/{today}/{now}"):
        os.mkdir(f"data/{today}/{now}")

    save_obj(parameters, "parameters", savepath)
    return savepath

def save_obj(obj, name, savepath):
    with open(savepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, filepath):
    with open(filepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Function to save weights
def save_weights(W, counter):
    today =  datetime.strftime(datetime.now(),'%y%m%d')
    now = datetime.strftime(datetime.now(),'%H%M%S')

    # Make sure folder is there
    if not os.path.isdir(f"figures/{today}/weights/"):
        os.mkdir(f"figures/{today}/weights/")

    csv_file = f"./figures/{today}/weights/{now}_{counter}.csv"

    # Note the saving if this is the first weight file
    if counter == 0:
        text_file = f"./figures/{today}/plot_log.txt"
        Path(text_file).touch()
        file = open(text_file,"a")
        file.write(f"At {now} a set of weights was saved\n")
        file.close()

    np.savetxt(csv_file, W)

def generate_rep(params, g):
    g = jnp.zeros(g.shape)
    g = g.at[:, 0].set(params[0])

    for i in range(g.shape[1]):
        g = g.at[:, i + 1].set(jnp.matmul(params[1], g[:, i]))

    return g