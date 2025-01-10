import wandb
import random
import torch
import os
import copy

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import pytz
from datetime import datetime
import utils
from absl import app, flags, logging

from data_gen.dataset import get_file_handler, export_data, MyDataset, InfiniteDataLooper

# import models 
import models_text_output
# from trainer import Trainer 
from trainer_text_output import Trainer
from tabulate import tabulate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import h5py

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
text_gen_filepath = f'text_generated_{timestamp}.txt'
text_test_filepath = f'text_test_{timestamp}.txt'

def eval_loss(trainer, samples, prefix, wandb_run):

    loss = trainer.get_loss(samples)

    print(f"train step: {trainer.train_step}, {prefix}_loss: {loss}")

    error_mean, error_std = trainer.get_error(samples)
    print(f"train step: {trainer.train_step}, {prefix}_error_mean: {error_mean}, {prefix}_error_std: {error_std}")
                    ### for vector valued fcts
                    # x = np.arange(len(error_mean))
                    # list_elements = [x]
                    # headers = [f"{prefix} example"]
                    # for cid in range(error_mean.shape[1]):
                    #     list_elements.append(error_mean[:, cid])
                    #     list_elements.append(error_std[:, cid])
                    #     headers.append(f"error mean, c:{cid}")
                    #     headers.append(f"error std, c:{cid}")
                    # table = list(zip(*list_elements))
                    # transpose = list(zip(*table))
                    # print(tabulate(table, headers=headers, tablefmt="grid"))


    if FLAGS.board:
        wandb_run.log({"step": trainer.train_step, f"{prefix}_loss": loss, f"{prefix}_error": error_mean})

def test_text_generation(trainer, samples, prefix, wandb_run, train = True):
    if FLAGS.board:
        # Specify the file path
        # file_path = 'text_generated.txt'
        # Append the generated text to the file
        if train:
            generated_text = trainer.test_text(samples, train = train)
            with open(text_gen_filepath, 'a') as file:
                file.write('Ground Truth: '+ samples["text"][0][2][0] + '\n')
                file.write('Generated text: '+ generated_text + '\n')  # Add a newline for separation
        
            # Save the file to the wandb run
            wandb.save(text_gen_filepath)
        else:
            generated_text_list = trainer.test_text(samples, train = train )
            with open(text_test_filepath, 'a') as file:
                for i in range(len(generated_text_list)):
                    file.write('Ground Truth: '+ samples["text"][i][2][0] + '\n')
                    file.write('Generated text: '+ generated_text_list[i] + '\n')  # Add a newline for separation
        
            # Save the file to the wandb run
            wandb.save(text_test_filepath)

def make_all_plots(trainer, samples, wandb_run):
    output, label, queries = trainer.get_plot_arrays(samples)  # output, label (bs, query_len, output_dim)

    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    
    for index in selected_indices:
        eq_type = samples['type'][index]
        dim = samples['dim'][index]
        
        # ODE Plotting (for dim == 1, 2, or 3)
        if dim in [1, 2, 3]:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            if dim == 1:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label')
                ax.legend()
                ax.set_title(f'ODE (1D) Eq type {eq_type}')
                
            elif dim == 2:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
                ax.legend()
                ax.set_title(f'ODE (2D) Eq type {eq_type}')
                
            elif dim == 3:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
                ax.plot(queries, output[index, :, 2], 'orange', label='Output z(t)')
                ax.plot(queries, label[index, :, 2], 'purple', linestyle='--', label='Label z(t)')
                ax.legend()
                ax.set_title(f'ODE (3D) Eq type {eq_type}')
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(output[index, :, 0], output[index, :, 1], output[index, :, 2], 'r-', label='Output')
                # ax.plot(label[index, :, 0], label[index, :, 1], label[index, :, 2], 'k--', label='Label')
                # ax.legend()
                # ax.set_title(f'ODE (3D) Eq type {eq_type}')
        
        # PDE Plotting (for dim > 3)
        elif dim > 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot label image in the first column
            im1 = axes[0].imshow(label[index], cmap='turbo', origin='lower')
            axes[0].set_title(f'Label PDE Eq. type {eq_type}')
            plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot output image in the second column
            im2 = axes[1].imshow(output[index], cmap='turbo', origin='lower')
            axes[1].set_title(f'Output PDE Eq. type {eq_type}')
            plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot absolute difference in the third column
            diff = np.abs(output[index] - label[index])
            im3 = axes[2].imshow(diff, cmap='turbo', origin='lower')
            axes[2].set_title(f'Abs Diff Eq. type {eq_type}')
            plt.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.0235, pad=0.04)
        
        # Adjust layout and save the plot with a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_key = f"output_label_plot_{timestamp}_index_{index}"
        plt.tight_layout()
        
        # Log the plot to wandb with a unique key
        wandb.log({unique_key: wandb.Image(plt)})
        
        # Close the plot to avoid memory issues
        plt.close()

    return

def make_all_test_plots(output, test_samples, queries, wandb_run):
    label = np.stack(test_samples['label'])
    output = output.detach().cpu().numpy()
    queries = queries.detach().cpu().numpy()

    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)

    for index in selected_indices:
        eq_type = test_samples['type'][index]
        dim = test_samples['dim'][index]

        # ODE Plotting (for dim == 1, 2, or 3)
        if dim in [1, 2, 3]:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            if dim == 1:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label')
                ax.legend()
                ax.set_title(f'ODE (1D) Eq type {eq_type}')
                
            elif dim == 2:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
                ax.legend()
                ax.set_title(f'ODE (2D) Eq type {eq_type}')
                
            elif dim == 3:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
                ax.plot(queries, output[index, :, 2], 'orange', label='Output z(t)')
                ax.plot(queries, label[index, :, 2], 'purple', linestyle='--', label='Label z(t)')
                ax.legend()
                ax.set_title(f'ODE (3D) Eq type {eq_type}')
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(output[index, :, 0], output[index, :, 1], output[index, :, 2], 'r-', label='Output')
                # ax.plot(label[index, :, 0], label[index, :, 1], label[index, :, 2], 'k--', label='Label')
                # ax.legend()
                # ax.set_title(f'ODE (3D) Eq type {eq_type}')

        # PDE Plotting (for dim > 3)
        elif dim > 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot label image in the first column
            im1 = axes[0].imshow(label[index], cmap='turbo', origin='lower')
            axes[0].set_title(f'Label PDE Eq. type {eq_type}')
            plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot output image in the second column
            im2 = axes[1].imshow(output[index], cmap='turbo', origin='lower')
            axes[1].set_title(f'Output PDE Eq. type {eq_type}')
            plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot absolute difference in the third column
            diff = np.abs(output[index] - label[index])
            im3 = axes[2].imshow(diff, cmap='turbo', origin='lower')
            axes[2].set_title(f'Abs Diff PDE Eq. type {eq_type}')
            plt.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.0235, pad=0.04)

        # Adjust layout and save the plot with a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_key = f"output_label_plot_{timestamp}_index_{index}"
        plt.tight_layout()

        # Log the plot to wandb with a unique key
        wandb.log({unique_key: wandb.Image(plt)})

        # Close the plot to avoid memory issues
        plt.close()

    return

def make_plots_extrapolate(output, test_samples, queries, wandb_run):
    label = np.stack(test_samples['label'])
    output = output.detach().cpu().numpy()
    queries = queries.detach().cpu().numpy()

    # Randomly select 4 indices to plot
    try:
        selected_indices = [i for i in range(output.shape[0])] #random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    relative_errors = []
    for index in selected_indices:
        eq_type = test_samples['type'][index]
        dim = test_samples['dim'][index]

        # ODE Plotting (for dim == 1, 2, or 3)
        if dim in [1, 2, 3]:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            if dim == 1:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output')
                ax.plot(queries, label[index, 64:, 0], 'k--', label='Label')
                ax.legend()
                ax.set_title(f'ODE (1D) Eq type {eq_type}')
                
            elif dim == 2:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, 64:, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, 64:, 1], 'b--', label='Label y(t)')
                ax.legend()
                ax.set_title(f'ODE (2D) Eq type {eq_type}')
                
            elif dim == 3:
                ax.plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
                ax.plot(queries, label[index, 64:, 0], 'k--', label='Label x(t)')
                ax.plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
                ax.plot(queries, label[index, 64:, 1], 'b--', label='Label y(t)')
                ax.plot(queries, output[index, :, 2], 'orange', label='Output z(t)')
                ax.plot(queries, label[index, 64:, 2], 'purple', linestyle='--', label='Label z(t)')
                ax.legend()
                ax.set_title(f'ODE (3D) Eq type {eq_type}')
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(output[index, :, 0], output[index, :, 1], output[index, :, 2], 'r-', label='Output')
                # ax.plot(label[index, :, 0], label[index, :, 1], label[index, :, 2], 'k--', label='Label')
                # ax.legend()
                # ax.set_title(f'ODE (3D) Eq type {eq_type}')

        # PDE Plotting (for dim > 3)
        elif dim > 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot label image in the first column
            im1 = axes[0].imshow(label[index, 64:], cmap='turbo', origin='lower')
            axes[0].set_title(f'Label PDE Eq. type {eq_type}')
            plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot output image in the second column
            im2 = axes[1].imshow(output[index], cmap='turbo', origin='lower')#, vmin =np.min(label[index, 64:]), vmax = np.max(label[index, 64:] ))
            axes[1].set_title(f'Output PDE Eq. type {eq_type}')
            plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.0235, pad=0.04)

            # Plot absolute difference in the third column
            diff = np.abs(output[index] - label[index, 64:])
            im3 = axes[2].imshow(diff, cmap='turbo', origin='lower')
            axes[2].set_title(f'Abs Diff PDE Eq. type {eq_type} index {index}')
            plt.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.0235, pad=0.04)
            error = np.sum(diff)/np.sum(label[index, 64:])
            relative_errors.append(error)
            print(index, "relative error = ",error)
        # Adjust layout and save the plot with a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_key = f"output_label_plot_{timestamp}_index_{index}"
        plt.tight_layout()

        # Log the plot to wandb with a unique key
        wandb.log({unique_key: wandb.Image(plt)})

        # Close the plot to avoid memory issues
        plt.close()
    print('mean relative error:', np.mean(relative_errors))
    return



def make_plots(trainer, samples, wandb_run):
    output, label, queries = trainer.get_plot_arrays(samples)  # output, label (bs, query_len, output_dim)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    for i, index in enumerate(selected_indices):
        row, col = divmod(i, 2)  # Divide the index to get the row and column position in the subplot
        # Plot in the current subplot
        eq_type = samples['type'][index]
        if samples['dim'][index] == 1:
            # print('output', output[index, :5, :])
            axes[row, col].plot(queries, output[index, :, 0], 'r-' , label='Output')
            axes[row, col].plot(queries, label[index, :, 0], 'k--' , label='Label')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')
        elif samples['dim'][index] == 2:
            axes[row, col].plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
            axes[row, col].plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
            axes[row, col].plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
            axes[row, col].plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')
        elif samples['dim'][index] == 3:
            axes[row, col].plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
            axes[row, col].plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
            axes[row, col].plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
            axes[row, col].plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
            axes[row, col].plot(queries, output[index, :, 2], 'orange', label='Output z(t)')
            axes[row, col].plot(queries, label[index, :, 2], 'purple', linestyle='--', label='Label z(t)')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')

            # 3D plot
            # ax = fig.add_subplot(2, 2, i+1, projection='3d')
            # ax.plot(output[index, :, 0], output[index, :, 1], output[index, :, 2],'r-' , label='Output')
            # ax.plot(label[index, :, 0], label[index, :, 1], label[index, :, 2], 'k--' , label='Label')
            # ax.legend()
            # ax.set_title(f'Eq type {eq_type}')
    # Adjust layout and save the plot with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_key = f"output_label_plot_{timestamp}"
    plt.tight_layout()

    # Log the plot to wandb with a unique key
    wandb.log({unique_key: wandb.Image(plt)})
    
    # Close the plot to avoid saving it again
    plt.close()
    return

def make_plots_PDE(trainer, samples, wandb_run):
    output, label, queries = trainer.get_plot_arrays(samples)  # output, label (bs, query_len, output_dim)
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))  # Adjusted to 3 columns
    
    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    
    for i, index in enumerate(selected_indices):
        eq_type = samples['type'][index]
        
        # Plot label image in the first column
        im1 = axes[i, 0].imshow(label[index], cmap='turbo', origin='lower')
        axes[i, 0].set_title(f'Label Eq. type {eq_type}')
        # axes[i, 0].axis('off')  # Hide axis
        
        # Add colorbar to the first column image
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar1.set_ticks([0, 0.5, 1])

        # Plot output image in the second column
        im2 = axes[i, 1].imshow(output[index], cmap='turbo', origin='lower')
        axes[i, 1].set_title(f'Output Eq. type {eq_type}')
        # axes[i, 1].axis('off')  # Hide axis
        
        # Add colorbar to the second column image
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar2.set_ticks([0, 0.5, 1])
        
        # Plot absolute difference in the third column
        diff = np.abs(output[index] - label[index])
        im3 = axes[i, 2].imshow(diff, cmap='turbo', origin='lower')
        axes[i, 2].set_title(f'Abs Diff Eq. type {eq_type}')
        # axes[i, 2].axis('off')  # Hide axis
        
        # Add colorbar to the third column image
        cbar3 = plt.colorbar(im3, ax=axes[i, 2], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar3.set_ticks([0, 0.5, 1])

    # Adjust layout and save the plot with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_key = f"output_label_plot_{timestamp}"
    plt.tight_layout()

    # Log the plot to wandb with a unique key
    wandb.log({unique_key: wandb.Image(plt)})
    
    # Close the plot to avoid saving it again
    plt.close()
    return

def make_test_plots(output, test_samples, queries, wandb_run):
    label = np.stack(test_samples['label'])
    output = output.detach().cpu().numpy()
    queries = queries.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    for i, index in enumerate(selected_indices):
        row, col = divmod(i, 2)  # Divide the index to get the row and column position in the subplot
        # Plot in the current subplot
        eq_type = test_samples['type'][index]
        if test_samples['dim'][index] == 1:
            # print('output', output[index, :5, :])
            axes[row, col].plot(queries, output[index, :, 0], 'r-' , label='Output')
            axes[row, col].plot(queries, label[index, :, 0], 'k--' , label='Label')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')
        elif test_samples['dim'][index] == 2:
            axes[row, col].plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
            axes[row, col].plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
            axes[row, col].plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
            axes[row, col].plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')
        elif test_samples['dim'][index] == 3:
            axes[row, col].plot(queries, output[index, :, 0], 'r-', label='Output x(t)')
            axes[row, col].plot(queries, label[index, :, 0], 'k--', label='Label x(t)')
            axes[row, col].plot(queries, output[index, :, 1], 'g-', label='Output y(t)')
            axes[row, col].plot(queries, label[index, :, 1], 'b--', label='Label y(t)')
            axes[row, col].plot(queries, output[index, :, 2], 'orange', label='Output z(t)')
            axes[row, col].plot(queries, label[index, :, 2], 'purple', linestyle='--', label='Label z(t)')
            axes[row, col].legend()
            axes[row, col].set_title(f'Eq type {eq_type}')
            # 3D plot
            # ax = fig.add_subplot(2, 2, i+1, projection='3d')
            # ax.plot(output[index, :, 0], output[index, :, 1], output[index, :, 2],'r-' , label='Output')
            # ax.plot(label[index, :, 0], label[index, :, 1], label[index, :, 2], 'k--' , label='Label')
            # ax.legend()
            # ax.set_title(f'Eq type {eq_type}')
    # Adjust layout and save the plot with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_key = f"output_label_plot_{timestamp}"
    plt.tight_layout()

    # Log the plot to wandb with a unique key
    wandb.log({unique_key: wandb.Image(plt)})
    
    # Close the plot to avoid saving it again
    plt.close()
    return

def make_test_plots_PDE(output, test_samples, queries, wandb_run):
    label = np.stack(test_samples['label'])
    output = output.detach().cpu().numpy()
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))  # Adjusted to 3 columns
    
    # Randomly select 4 indices to plot
    try:
        selected_indices = random.sample(range(output.shape[0]), 4)
    except:
        selected_indices = random.choices(range(output.shape[0]), k=4)
    
    for i, index in enumerate(selected_indices):
        eq_type = test_samples['type'][index]
        
        # Plot label image in the first column
        im1 = axes[i, 0].imshow(label[index], cmap='turbo',  origin='lower')
        axes[i, 0].set_title(f'Label Eq. type {eq_type}')
        # axes[i, 0].axis('off')  # Hide axis
        
        # Add colorbar to the first column image
        cbar1 = plt.colorbar(im1, ax=axes[i, 0], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar1.set_ticks([0, 0.5, 1])

        # Plot output image in the second column
        im2 = axes[i, 1].imshow(output[index][:], cmap='turbo', origin='lower')
        axes[i, 1].set_title(f'Output Eq. type {eq_type}')
        # axes[i, 1].axis('off')  # Hide axis
        
        # Add colorbar to the second column image
        cbar2 = plt.colorbar(im2, ax=axes[i, 1], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar2.set_ticks([0, 0.5, 1])
        
        # Plot absolute difference in the third column
        diff = np.abs(output[index][:] - label[index])
        im3 = axes[i, 2].imshow(diff, cmap='turbo', origin='lower')
        axes[i, 2].set_title(f'Abs Diff Eq. type {eq_type}')
        # axes[i, 2].axis('off')  # Hide axis
        
        # Add colorbar to the third column image
        cbar3 = plt.colorbar(im3, ax=axes[i, 2], orientation='vertical', fraction=0.0235, pad=0.04)
        # cbar3.set_ticks([0, 0.5, 1])

    # Adjust layout and save the plot with a unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_key = f"output_label_plot_{timestamp}"
    plt.tight_layout()

    # Log the plot to wandb with a unique key
    wandb.log({unique_key: wandb.Image(plt)})
    
    # Close the plot to avoid saving it again
    plt.close()
    return

def make_image(array, wandb=True, title=None, config=None):
    fig = plt.figure(figsize=(4, 4))
    cmap = "bwr"
    vmax = np.max(np.abs(array))
    plt.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if not wandb:
        return fig
    wandb_image = utils.plt_to_wandb(fig, config)
    plt.close("all")
    return wandb_image


def run_train():
    utils.set_seed(FLAGS.seed)
    tc_rng = torch.Generator()
    tc_rng.manual_seed(FLAGS.seed)

    # prepare configs
    flag_config = {value.name: value._value for key, value in FLAGS.__flags.items()}

    train_data_config = utils.load_json("config_data/" + FLAGS.train_data_config)
    test_data_config = utils.load_json("config_data/" + FLAGS.test_data_config)

    train_data_config["folder"] = FLAGS.data_home_folder + train_data_config["folder"]
    test_data_config["folder"] = FLAGS.data_home_folder + test_data_config["folder"]

    model_config = utils.load_json("config_model/" + FLAGS.model_config)

    train_warmup_steps = max(1, FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_warmup_percent // 100)
    train_decay_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_decay_percent // 100
    opt_config = {
        "peak_lr": FLAGS.train_peak_lr,
        "end_lr": FLAGS.train_end_lr,
        "warmup_steps": train_warmup_steps,
        "decay_steps": train_decay_steps,
        "gnorm_clip": FLAGS.train_gnorm_clip,
        "weight_decay": FLAGS.train_weight_decay,
    }

    train_data = MyDataset(
        train=True,
        config=train_data_config,
        FLAGS=FLAGS,
        size=FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_batch_size if FLAGS.export_data else None,
    )
    test_data = MyDataset(
        train=False,
        config=test_data_config,
        FLAGS=FLAGS,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=FLAGS.train_batch_size,
        num_workers=FLAGS.dataset_workers,
        collate_fn=train_data.collate_fn,
        shuffle=not FLAGS.export_data,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=FLAGS.test_batch_size,
        num_workers=FLAGS.dataset_workers,
        collate_fn=test_data.collate_fn,
        shuffle=not FLAGS.export_data,
        pin_memory=True,
    )

    all_configs = {
        "flag_config": flag_config,
        "train_data_config": train_data_config,
        "test_data_config": test_data_config,
        "model_config": model_config,
        "opt_config": opt_config,
    }

    if FLAGS.board:
        wandb_run = wandb.init(
            project=FLAGS.project,
            config=all_configs,
        )
    else:
        wandb_run = None

    if FLAGS.export_data:
        trainer = None
    else:
        model = models_text_output.ModelWrapper(model_config)
        trainer = Trainer(model, model_config, opt_config, trainable_mode=FLAGS.trainable_mode, t_start=FLAGS.t_start, t_end=FLAGS.t_end, t_len=FLAGS.t_len, amp=FLAGS.amp)

    # some prints
    print("\n")
    time_stamp = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    stamp = time_stamp
    print("stamp: {}".format(stamp))

    for key, value in all_configs.items():
        print("-" * 20 + key + "-" * 20)
        print(value)
        print()

    print("\n")

    # out = next(iter(test_loader))
    # for o in out:
    #     print(o.shape)
    # trainer.summary(out[1])

    # training

    train_looper = InfiniteDataLooper(train_loader)
    test_looper = InfiniteDataLooper(test_loader)

    if FLAGS.export_data:

        if FLAGS.export_data_type == "train":
            save_file = get_file_handler(train_data_config["folder"] + train_data_config["text_filename"])
            save_data_path = train_data_config["folder"] + train_data_config["data_filename"]
        else:
            save_file = get_file_handler(test_data_config["folder"] + test_data_config["text_filename"])
            save_data_path = test_data_config["folder"] + test_data_config["data_filename"]

        total_lines = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_batch_size
        print('total lines', total_lines)
        cur_line = 0

        save_data_matrix = np.zeros(
            (total_lines, FLAGS.t_len, 1 + model_config["max_dimension"]), #or 1+model_config["output_dimension"]
            dtype=np.single,
        ) #for PDES (total_lines, FLAGS.t_len, FLAGS.spaces, 1 + model_config["max_dimension"])
        save_control_matrix = np.zeros(
            (total_lines, FLAGS.t_len, 1 + model_config["max_dimension"]), #or 1+model_config["output_dimension"]
            dtype=np.single,
        )
        save_coeff_matrix = np.zeros(
            (total_lines,  model_config["max_number_coeffs"]),
            dtype=np.single,
        )

    total_steps = FLAGS.epochs * FLAGS.steps_per_epoch

    for step in range(1, total_steps + 1):
        samples = next(train_looper)

        if FLAGS.export_data:
            export_data(save_file, samples, save_data_matrix, save_control_matrix, save_coeff_matrix, cur_line)
            cur_line += FLAGS.train_batch_size
            continue

        if (
            (trainer.train_step % FLAGS.loss_freq == 0)
            or (trainer.train_step % (FLAGS.loss_freq // 10) == 0 and trainer.train_step <= FLAGS.loss_freq)
            or (
                trainer.train_step % (FLAGS.loss_freq // 10) == 0
                and trainer.train_step >= total_steps - FLAGS.loss_freq
            )
        ):
            # log every loss_freq step, and more frequently in the start and end
            test_samples = next(test_looper)
            eval_loss(trainer, samples, "train", wandb_run)
            eval_loss(trainer, test_samples, "test", wandb_run)
            test_text_generation(trainer, test_samples,"test", wandb_run)

        # if (
        #     (trainer.train_step % FLAGS.plot_freq == 0)
        #     or (trainer.train_step % (FLAGS.plot_freq // 10) == 0 and trainer.train_step <= FLAGS.plot_freq)
        #     or (
        #         trainer.train_step % (FLAGS.plot_freq // 10) == 0
        #         and trainer.train_step >= total_steps - FLAGS.plot_freq
        #     )
        # ):
        #     test_samples = next(test_looper)
        #     eval_plot(trainer, test_samples, "test", in_idx, out_idx, 0, wandb_run)  # 0 as bid
        #     eval_plot(trainer, samples, "train", train_in_idx, train_out_idx, 0, wandb_run)  # 0 as bid
        if FLAGS.board and (trainer.train_step % (FLAGS.save_freq) == 0 or step == total_steps):#trainer.train_step % (FLAGS.save_freq) == 0 or step == total_steps:
            print("Saving current model...")
            ckpt_dir = f"../ckpts/{FLAGS.project}/" + stamp
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            print("current time: " + datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S"))
            trainer.save(ckpt_dir)
            
        if (step % (10*FLAGS.steps_per_epoch) == 0) or step == total_steps:
            #Plot test trajectory after training
            if FLAGS.dataset == "PDE":
                make_plots_PDE(trainer, test_samples, wandb_run)
            elif FLAGS.dataset == "ODE":
                make_plots(trainer, test_samples, wandb_run)
            else:
                make_all_plots(trainer, test_samples, wandb_run)
    
        trainer.iter(samples)

        if trainer.train_step == FLAGS.time_warm:  # exclude warming up steps
            utils.timer.tic("time estimate")
        if trainer.train_step > 0 and (trainer.train_step % FLAGS.time_freq == 0):
            ratio = (trainer.train_step - FLAGS.time_warm) / (FLAGS.epochs * FLAGS.steps_per_epoch)
            samples_processed = (trainer.train_step - FLAGS.time_warm) * FLAGS.train_batch_size
            utils.timer.estimate_time("time estimate", ratio, samples_processed)

    
    if FLAGS.export_data:
        print(f"final line: {cur_line}")
        with h5py.File(save_data_path, "w") as hf:
            hf.create_dataset("data", data=save_data_matrix, maxshape=(None, FLAGS.t_len,1+model_config["max_dimension"])) #changed maxshape for 3D cse
            hf.create_dataset("control", data=save_control_matrix, maxshape=(None, FLAGS.t_len,1+ model_config["max_dimension"])) #changed maxshape for 3D cse
            hf.create_dataset("coefficients", data=save_coeff_matrix)
        print(f"Data have been stored in h5 in: {save_data_path}.")

    if FLAGS.board:
        wandb.finish()


def run_test():
    # prepare configs
    flag_config = {value.name: value._value for key, value in FLAGS.__flags.items()}

    test_data_config = utils.load_json("config_data/test_data_config.json")
    test_data_config["folder"] = FLAGS.data_home_folder + test_data_config["folder"]

    model_config = utils.load_json("config_model/" + FLAGS.model_config)
    train_warmup_steps = max(1, FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_warmup_percent // 100)
    train_decay_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_decay_percent // 100
    opt_config = {
        "peak_lr": FLAGS.train_peak_lr,
        "end_lr": FLAGS.train_end_lr,
        "warmup_steps": train_warmup_steps,
        "decay_steps": train_decay_steps,
        "gnorm_clip": FLAGS.train_gnorm_clip,
        "weight_decay": FLAGS.train_weight_decay,
    }

    test_data = MyDataset(
        train=False,
        config=test_data_config,
        FLAGS=FLAGS,
        size=FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.test_batch_size if FLAGS.export_data else None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=FLAGS.test_batch_size,
        num_workers=FLAGS.dataset_workers,
        collate_fn=test_data.collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    all_configs = {
        "flag_config": flag_config,
        "test_data_config": test_data_config,
        "model_config": model_config,
        "opt_config": opt_config,
    }

    if FLAGS.board:
        wandb_run = wandb.init(
            project=FLAGS.project,
            config=all_configs,
        )
    else:
        wandb_run = None

    if FLAGS.export_data:
        trainer = None
    else:
        model = models_text_output.ModelWrapper(model_config)
        trainer = Trainer(model, model_config, opt_config, trainable_mode=FLAGS.trainable_mode, t_start=FLAGS.t_start, t_end=FLAGS.t_end, t_len=FLAGS.t_len, amp=FLAGS.amp)

    test_looper = InfiniteDataLooper(test_loader)
    
    if FLAGS.export_data:
        save_file = get_file_handler(test_data_config["folder"] + test_data_config["text_filename"])
        save_data_path = test_data_config["folder"] + test_data_config["data_filename"]

        total_lines = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.test_batch_size
        print('total lines', total_lines)
        cur_line = 0

        save_data_matrix = np.zeros(
            (total_lines, FLAGS.t_len, 1 + model_config["max_dimension"]), #or 1+model_config["output_dimension"]
            dtype=np.single,
        )
        save_control_matrix = np.zeros(
            (total_lines, FLAGS.t_len, 1 + model_config["max_dimension"]), #or 1+model_config["output_dimension"]
            dtype=np.single,
        )
        save_coeff_matrix = np.zeros(
            (total_lines,  model_config["max_number_coeffs"]),
            dtype=np.single,
        )
        total_steps = FLAGS.epochs * FLAGS.steps_per_epoch

        for step in range(1, total_steps + 1):
            samples = next(test_looper)
            export_data(save_file, samples, save_data_matrix, save_control_matrix, save_coeff_matrix, cur_line)
            cur_line += FLAGS.test_batch_size

    
        print(f"final line: {cur_line}")
        with h5py.File(save_data_path, "w") as hf:
            hf.create_dataset("data", data=save_data_matrix, maxshape=(None, FLAGS.t_len,1+model_config["max_dimension"])) #changed maxshape for 3D cse
            hf.create_dataset("control", data=save_control_matrix, maxshape=(None, FLAGS.t_len,1+ model_config["max_dimension"])) #changed maxshape for 3D cse
            hf.create_dataset("coefficients", data=save_coeff_matrix)
        print(f"Data have been stored in h5 in: {save_data_path}.")
    else:
        # Load model
        data_mod = torch.load(test_data_config['model_folder'] + FLAGS.svd_model_folder)
        weights = data_mod["model"]
        model.load_state_dict(weights)
        
        # Initialize accumulators for total test error and total standard deviation
        total_error_sum = 0.0
        total_std_sum = 0.0
        total_samples = 0
        plot_count = 0

        # Testing loop for calculating total test error and std
        for test_samples in test_loader:
            queries = torch.linspace(FLAGS.t_start, FLAGS.t_end, FLAGS.t_len, dtype=torch.float)[1:].cuda()
            output = model.number_test_output(test_samples, queries)
            
            # Get batch error mean and std
            error_mean, error_std = trainer.get_error(test_samples)
            batch_size = len(test_samples['data'])
            
            # Accumulate weighted error mean and std
            total_error_sum += error_mean * batch_size
            total_std_sum += error_std * batch_size
            total_samples += batch_size

            # Plot results per batch
            if (FLAGS.dataset == "PDE" and plot_count < 5):
                make_test_plots_PDE(output, test_samples, queries, wandb_run)
                plot_count += 1
            elif (FLAGS.dataset == "ODE" and plot_count < 5):
                make_test_plots(output, test_samples, queries, wandb_run)
                plot_count += 1
            elif (FLAGS.dataset == "both" and plot_count < 5):
                make_all_test_plots(output, test_samples, queries, wandb_run)
                plot_count += 1

        # Calculate average test error and std
        average_test_error = total_error_sum / total_samples
        average_test_std = total_std_sum / total_samples
        print(f"Average Test Error: {average_test_error}")
        print(f"Average Test Std Dev: {average_test_std}")
        # Text generation for the batch
        test_text_generation(trainer, test_samples, "test", wandb_run, train=False)

    if FLAGS.board:
        wandb.finish()

def run_extrap():
        # prepare configs
    flag_config = {value.name: value._value for key, value in FLAGS.__flags.items()}

    test_data_config = utils.load_json("config_data/test_data_config.json")
    test_data_config["folder"] = FLAGS.data_home_folder + test_data_config["folder"]

    model_config = utils.load_json("config_model/" + FLAGS.model_config)
    train_warmup_steps = max(1, FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_warmup_percent // 100)
    train_decay_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.train_decay_percent // 100
    opt_config = {
        "peak_lr": FLAGS.train_peak_lr,
        "end_lr": FLAGS.train_end_lr,
        "warmup_steps": train_warmup_steps,
        "decay_steps": train_decay_steps,
        "gnorm_clip": FLAGS.train_gnorm_clip,
        "weight_decay": FLAGS.train_weight_decay,
    }

    test_data = MyDataset(
        train=False,
        config=test_data_config,
        FLAGS=FLAGS,
        size=FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.test_batch_size if FLAGS.export_data else None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=FLAGS.test_batch_size,
        num_workers=FLAGS.dataset_workers,
        collate_fn=test_data.collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    all_configs = {
        "flag_config": flag_config,
        "test_data_config": test_data_config,
        "model_config": model_config,
        "opt_config": opt_config,
    }

    if FLAGS.board:
        wandb_run = wandb.init(
            project=FLAGS.project,
            config=all_configs,
        )
    else:
        wandb_run = None
    
    model = models_text_output.ModelWrapper(model_config)
    trainer = Trainer(model, model_config, opt_config, trainable_mode=FLAGS.trainable_mode, t_start=FLAGS.t_start, t_end=FLAGS.t_end, t_len=FLAGS.t_len, amp=FLAGS.amp)
    test_looper = InfiniteDataLooper(test_loader)

    # Load model
    data_mod = torch.load(test_data_config['model_folder'] + FLAGS.svd_model_folder)
    weights = data_mod["model"]
    model.load_state_dict(weights)

    #extrapolate for one batch
    test_samples = next(test_looper)

    #get queries and current output
    queries = torch.linspace(FLAGS.t_start, FLAGS.t_end, FLAGS.t_len, dtype=torch.float)[1:].cuda()
    output = model.number_test_output(test_samples, queries)

    #generate new input, initial condition is predicted solution at end time.
    test_samples2 = copy.deepcopy(test_samples)
    last_time_sol = output[:, -1:, :]
    print(output.shape[0])
    initial_time = torch.full((output.shape[0], 1, 1), FLAGS.t_start).cuda()
    # Concatenate the column with the new intitial time tensor along the last dimension
    expanded_tensor = torch.cat((initial_time, last_time_sol), dim=2)  # Shape: (4, 1, 129)
    # Convert the tensor into a list with b.s. elements, each of size (1, 129)
    new_IC = [expanded_tensor[i] for i in range(expanded_tensor.size(0))]
    test_samples2['data'] = new_IC
    queries2 = torch.linspace(FLAGS.t_start, FLAGS.t_end, FLAGS.t_len, dtype=torch.float)[1:].cuda()
    output2 = model.number_test_output(test_samples2, queries2)
    make_plots_extrapolate(output2, test_samples2, queries2, wandb_run)


def main(argv):
    if FLAGS.dry_run:
        FLAGS.epochs = 2
        FLAGS.steps_per_epoch = 50
        FLAGS.loss_freq = 10

    if FLAGS.export_data:
        FLAGS.epochs = 1

    if FLAGS.main == "train":
        run_train()
    elif FLAGS.main == "test":
        run_test()
    elif FLAGS.main == "extrapolate":
        run_extrap()


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_boolean("board", False, "log data online through wanbd")
    flags.DEFINE_boolean("amp", False, "using automatic mixed precision")
    flags.DEFINE_boolean("export_data", False, "generate data and export to disk")
    flags.DEFINE_string("export_data_type", "train", "type of exported data")

    flags.DEFINE_boolean("dry_run", False, "run a quick test")

    flags.DEFINE_boolean("deterministic", True, "deterministic mode")

    flags.DEFINE_string("user", "yl", "user name, used for saving results and check points")
    flags.DEFINE_string("project", "test", "project")

    flags.DEFINE_integer("seed", 87, "random seed")

    flags.DEFINE_integer("profile_level", 0, "0: usual training, 1: profile training, 2: profile without loading data")

    flags.DEFINE_string("data_home_folder", "/home/elisa/code/icon-gen/dataset/", "folder for training data")
    flags.DEFINE_string("svd_model_folder", "20240712-171318/299999_params.pth", "folder for saved model for testing")

    flags.DEFINE_string("train_data_config", "train_data_config.json", "config file for training")
    flags.DEFINE_string("test_data_config", "valid_data_config.json", "config file for testing")
    flags.DEFINE_integer("dataset_workers", 2, "number of workers for dataset")

    flags.DEFINE_string("model", "icon", "model name")
    flags.DEFINE_string("model_config", "model.json", "config file for model")

    flags.DEFINE_string("restore_dir", None, "restore directory")
    flags.DEFINE_integer("restore_step", 1000000, "restore step")
    flags.DEFINE_string("trainable_mode", "all", "trainable variables")

    flags.DEFINE_integer("train_batch_size", 6, "batch size")
    flags.DEFINE_integer("test_batch_size", 6, "test batch size")

    flags.DEFINE_float("train_peak_lr", 0.0001, "training peak learning rate")
    flags.DEFINE_float("train_end_lr", 0.0, "training ending learning rate")
    flags.DEFINE_integer("train_warmup_percent", 10, "training warmup percentage")
    flags.DEFINE_integer("train_decay_percent", 100, "training decay percentage")
    flags.DEFINE_float("train_gnorm_clip", 1.0, "training gradient global norm clip")
    flags.DEFINE_float("train_weight_decay", 0.0001, "training weight decay")

    flags.DEFINE_integer("epochs", 20, "total num of epochs")
    flags.DEFINE_integer("steps_per_epoch", 10000, "steps per epoch")

    flags.DEFINE_integer("loss_freq", 1000, "frequency of printing loss")
    flags.DEFINE_integer("save_freq", 100000, "frequency of saving model")
    flags.DEFINE_integer("plot_freq", 10000, "frequency of plotting to board")
    flags.DEFINE_integer("time_freq", 1000, "frequency of estimating time")
    flags.DEFINE_integer("time_warm", 100, "warming up steps for timing")
    flags.DEFINE_integer("plot_num", None, "number of plot cases to board")

    flags.DEFINE_enum("main", "train", ["test", "train", "extrapolate"], "train or test or estrapolate")

    flags.DEFINE_integer("t_len", 100, "total number of times to sample solution and control, query length will be t_len-1.")
    flags.DEFINE_float("t_start", 0, "time interval start")
    flags.DEFINE_float("t_end", 3, "time interval end")

    flags.DEFINE_list("sentence_ids", [0,1,2,3,4,5],"equation types to use")

    # flags.DEFINE_boolean("PDE", False, "Using PDE dataset")
    flags.DEFINE_enum("dataset", "both", ["ODE", "PDE", "both"], "which dataset are we using?")

    flags.DEFINE_integer("IC_per_eq", 100, "Number of initial conditions per equation/coefficient.")
    flags.DEFINE_string("IC_types", "train", "type of IC for PDE")

    app.run(main)
