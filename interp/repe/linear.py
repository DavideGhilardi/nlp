import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch

from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

def subspace_ablation_hook(
        rs: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        lam: float,
        subspace: Float[torch.Tensor, "d_model n_comp"],
        mean_rs: Float[torch.Tensor, "d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:

        P_u = subspace.to(rs.device).type(rs.dtype) @ subspace.to(rs.device).type(rs.dtype).T #d_mod, d_mod
        rs = rs + (lam * (mean_rs.to(rs.device) - rs) @ P_u)

        return rs

def plot_pc(activations, labels, prompts=None, generations=None, blocked=None, highlight_indices=None, palette=None):
    if palette is None:
        palette=['#2a9d8f', '#e76f51'] 
    df = pd.DataFrame({
        '1PC': activations[:, 0],
        '2PC': activations[:, 1],
        'label': labels
    })

    hover_data = ['index']

    if prompts is not None:
        df['Input'] = prompts
        hover_data.append('Input')


    if generations is not None:
        df['Output'] = [g[:128] for g in generations]
        hover_data.append('Output')

    # Creating the scatter plot with the custom color map
    fig = px.scatter(df.reset_index(), x='1PC', y='2PC', color='label',
                     hover_data=hover_data, opacity=0.7, symbol=blocked,
                     symbol_sequence= ['circle', 'cross'],
                     color_discrete_sequence=palette, title=f'First 2 PCs of activations')


    if highlight_indices is not None:
        highlight_df = df.loc[highlight_indices]
        fig.add_trace(px.scatter(highlight_df.reset_index(), x='1PC', y='2PC', 
                                 color_discrete_sequence=['yellow'], 
                                 hover_data=hover_data,
                                ).update_traces(marker=dict(size=12, opacity=1, line=dict(width=2, color='DarkSlateGrey'))).data[0])
    
    return fig

"""
def plot_pc(df, name, highlight_indices=None):
    palette = ['#73a942', '#ef233c', '#0077b6'] 
    # Creating the scatter plot with the custom color map
    fig = px.scatter(df.reset_index(), x='1PC', y='2PC', color='label', symbol=f'Blocked_{rs}',
                     hover_data=['index', 'Instruction', f'Answer_{rs}', f'Blocked_{rs}'], opacity=0.7, 
                     color_discrete_sequence=palette, symbol_sequence=['circle', 'x'], title=f'First 2 PCs of activations - {name}')


    # Highlight specific points if indices are provided
    if highlight_indices is not None:
        # Extract the rows to highlight based on provided indices
        highlight_df = df.loc[highlight_indices]
        # Add highlighted points as a separate trace with different marker properties
        fig.add_trace(px.scatter(highlight_df.reset_index(), x='1PC', y='2PC', 
                                 color_discrete_sequence=['yellow'], # Color for highlighted points
                                 hover_data=['index', 'Instruction', f'Answer_{rs}', f'Blocked_{rs}'],
                                ).update_traces(marker=dict(size=12, opacity=1, line=dict(width=2, color='DarkSlateGrey'))).data[0])
        
    # Set custom labels for the legend
    
    # Update the layout size
    fig.update_layout(width=1200, height=800)

    # Show the figure
    fig.show()


def plot_pc_3d(df, name):
    palette = sns.color_palette(['#73a942', '#ef233c', '#0077b6'])
    # Creating the 3D scatter plot with the custom color map
    fig = px.scatter_3d(df, x='1PC', y='2PC', z='3PC', color='label', symbol='Blocked',
                        hover_data=['Instruction', 'Answer', 'Blocked'], opacity=0.7, symbol_sequence=['circle', 'x'],
                        color_discrete_sequence=palette, title=f'First 3 PCs of activations - {name}')
    
    # Customize marker settings
    fig.update_traces(marker=dict(size=3))  # Adjust size as needed

    # Update the layout size and axis titles
    fig.update_layout(width=1200, height=800, scene=dict(
                        xaxis_title='1PC',
                        yaxis_title='2PC',
                        zaxis_title='3PC'))

    # Show the figure
    fig.show()
"""