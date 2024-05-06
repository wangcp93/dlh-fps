import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import pandas as pd

from io_utils import load_data
from build_vanilla_net import build_morgan_deep_net
from util import normalize_array, build_batched_grad, rmse
from optimizers import adam



def train_nn(pred_fun, loss_fun, num_weights, train_smiles, train_raw_targets, train_params, 
             seed=0, validation_smiles=None, validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    num_print_examples = 100
    train_targets, undo_norm = normalize_array(train_raw_targets)
    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            train_preds = undo_norm(pred_fun(weights, train_smiles[:num_print_examples]))
            cur_loss = loss_fun(weights, train_smiles[:num_print_examples], train_targets[:num_print_examples])
            training_curve.append(cur_loss)
            print("Iteration", iter, "loss", cur_loss,\
                  "train RMSE", rmse(train_preds, train_raw_targets[:num_print_examples]), end=' ')
            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print("Validation RMSE", iter, ":", rmse(validation_preds, validation_raw_targets))

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)

    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve



def run_ECFP_experiment(train_df, val_df, test_df, fp_depth):
    
    train_inputs = train_df["smiles"].tolist()
    train_targets = train_df["label"].tolist()
    val_inputs = val_df["smiles"].tolist()
    val_targets = val_df["label"].tolist()
    test_inputs = test_df["smiles"].tolist()
    test_targets = test_df["label"].tolist()

    # Define model architecture for building morgan fingerprint
    model_params = dict(fp_length=2048,    # Usually neural fps need far fewer dimensions than morgan.
                        fp_depth=fp_depth,      # The depth of the network equals the fingerprint radius.
                        conv_width=50,   # Only the neural fps need this parameter.
                        h1_size=1024,     # Size of hidden layer of network on top of fps.
                        L2_reg=np.exp(-8))
    train_params = dict(num_iters=100,
                        batch_size=100,
                        init_scale=np.exp(-4),
                        step_size=np.exp(-6))

    # Define the architecture of the network that sits on top of the fingerprints.
    vanilla_net_params = dict(
        layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
        normalize=True, L2_reg = model_params['L2_reg'], nll_func = rmse)
    

    def print_performance(pred_func):
        train_preds = pred_func(train_inputs)
        val_preds = pred_func(val_inputs)
        print("\nPerformance (RMSE):")
        print("Train:", rmse(train_preds, train_targets))
        print("Test: ", rmse(val_preds,  val_targets))
        print("-" * 80)
        return rmse(val_preds, val_targets)

    def run_morgan_experiment():
        loss_fun, pred_fun, net_parser = \
            build_morgan_deep_net(model_params['fp_length'],
                                  model_params['fp_depth'], vanilla_net_params)
        num_weights = len(net_parser)
        predict_func, trained_weights, conv_training_curve = \
            train_nn(pred_fun, loss_fun, num_weights, train_inputs, train_targets, train_params,
                     validation_smiles=val_inputs, validation_raw_targets=val_targets)
        return print_performance(predict_func)

    test_loss_morgan = run_morgan_experiment()
    return test_loss_morgan
