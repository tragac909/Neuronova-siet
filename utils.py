import numpy as np
import matplotlib.pyplot as plt
import os


def press_to_quit(e):
    if e.key in {'q', 'escape'}:
        os._exit(0)  # unclean exit, but works
    if e.key in {' ', 'enter'}:
        plt.close()  # skip blocking figures


def show_history(history, block=True):
    fig = plt.figure(num='Training history')
    fig.canvas.mpl_connect('key_press_event', press_to_quit)
    plt.title('Loss per epoch')
    if 'loss' in history:
        plt.plot(history['loss'], '-b', label='training loss')
    else:
        print("[WARNING] 'loss' key not found in history")
    if 'val_loss' in history:
        plt.plot(history['val_loss'], '-r', label='validation loss')

    plt.grid(True)
    plt.legend(loc='best')
    plt.xlim(left=-1)
    plt.ylim(bottom=-0.01)
    plt.tight_layout()
    plt.show(block=block)


def show_data(X, y, predicted=None, s=30, block=True):
    plt.figure(num='Data', figsize=(9, 9)).canvas.mpl_connect('key_press_event', press_to_quit)
    if predicted is not None:
        predicted = np.asarray(predicted).flatten()
        plt.subplot(2, 1, 2)
        plt.title('Predicted')
        # print('s',10 + s * max(0, max(predicted)))
        sc = plt.scatter(X[0, :], X[1, :],
                    c=predicted, cmap='coolwarm',
                    s=10 + s * np.maximum(0, max(predicted)))
        plt.colorbar(sc, label="Predicted Values")  # Add simple colorbar
        plt.subplot(2, 1, 1)
        plt.title('Original')

    y = np.asarray(y).flatten()
    sc2 = plt.scatter(X[0,:], X[1,:],
                c=y, cmap='coolwarm',
                s=10 + s * np.maximum(0, max(predicted)))
    plt.colorbar(sc2, label="Predicted Values")  # Add simple colorbar
    plt.tight_layout()
    plt.show(block=block)
