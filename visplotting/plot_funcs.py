import pandas as pd
import matplotlib.pyplot as plt


# calculate statistics and plot histogram for the label distribution
def print_label_statistics(data_df, target_name):
    label = data_df["label"]
    
    # Generate statistics
    print("Count:", len(label))
    print("Mean:", f"{label.mean():.2f}")
    print("Stdev:", f"{label.std():.2f}")
    print("Median:", f"{label.median():.2f}")
    print("Max:", f"{label.max():.2f}")
    print("Min:", f"{label.min():.2f}")
    
    # Generate histogram
    plt.figure(figsize = (4, 3))
    label.plot.hist(bins = 10)
    plt.title(f'Histogram of {target_name}')
    plt.xlabel(target_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Plot training loss over iterations
def plot_train_loss(train_losses):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_losses, marker='', linestyle='-')
    plt.title('Training loss over the number of training epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.grid(True)
    plt.show()


# Plot training and validation error over iterations
def plot_train_val_error(train_err, val_err):
    epochs = list(range(1, len(train_err) + 1))
    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_err, marker='', linestyle='-', label = "Train RMSE")
    plt.plot(epochs, val_err, marker='', linestyle='-', label = "Val RMSE")
    plt.title('RMSE over the number of training epochs')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    plt.show()


# Plot rmse of different models over fingerprint depth (number of layers)
def plot_rmse_over_layers(good_nfp, bad_nfp, ecfp, good_legend, bad_legend):
    fp_depths = list(range(1, len(good_nfp) + 1))
    ymax = max(*good_nfp, *bad_nfp, *ecfp)
    ymin = min(*good_nfp, *bad_nfp, *ecfp)
    yspan = ymax - ymin
    
    plt.figure(figsize=(4, 3))
    plt.plot(fp_depths, good_nfp, marker='', linestyle='-', label = good_legend)
    plt.plot(fp_depths, bad_nfp, marker='', linestyle='-', label = bad_legend)
    plt.plot(fp_depths, ecfp, marker='', linestyle='-', label = 'Circular fp')
    plt.title('Test RMSE over fingerprint depth on Solubility dataset')
    plt.xlabel('Fingerprint depth (network layers)')
    plt.ylabel('RMSE')
    plt.ylim(top = ymax + yspan / 4)
    plt.grid(True)
    plt.legend(fontsize = "small")
    plt.show()
