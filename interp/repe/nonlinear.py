from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

def plot_som(som, activations, labels, colors=None, alpha=.2):    
    w_x, w_y = zip(*[som.winner(d) for d in activations])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    fig = plt.figure(figsize=(10, 6), dpi=150)
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    if colors is None:
        colors = sns.color_palette('Blues', len(labels))

    for i, c in enumerate(np.unique(labels)):
        idx_target = t==c
        plt.scatter(w_x[idx_target] +.2+(np.random.rand(np.sum(idx_target))-.5)*.4,
                    w_y[idx_target] +.2+(np.random.rand(np.sum(idx_target))-.5)*.4, 
                    s=20, c=colors[i], label=c)
        plt.legend(loc='upper right')
        plt.grid()

    return fig

def plot_som_plotly(som, activations, labels, prompts=None, generations=None, blocked=None, highlight_indices=None, palette=None, alpha=.2):
    if palette is None:
        palette = ['#73a942', '#ef233c'] 
    
    w_x, w_y = zip(*[som.winner(d) for d in activations])
    if type(w_x[0]) == torch.Tensor:
        w_x = torch.stack(w_x).cpu()
        w_y = torch.stack(w_y).cpu()

    df = pd.DataFrame({
        'X': np.array(w_x) + (np.random.rand(len(activations)) - 0.5) * 0.4, 
        'Y': np.array(w_y) + (np.random.rand(len(activations)) - 0.5) * 0.4, 
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
    fig = px.scatter(df.reset_index(), x='X', y='Y', color='label',
                     hover_data=hover_data, opacity=0.7, symbol=blocked,
                     symbol_sequence= ['circle', 'cross'],
                     color_discrete_sequence=palette)

    if highlight_indices is not None:
        highlight_df = df.loc[highlight_indices]
        fig.add_trace(px.scatter(highlight_df.reset_index(), x='X', y='Y', 
                                 color_discrete_sequence=['yellow'], 
                                 hover_data=hover_data,
                                ).update_traces(marker=dict(size=12, opacity=1, line=dict(width=2, color='DarkSlateGrey'))).data[0])
        
    fig.add_trace(go.Heatmap(z=som.distance_map().T, colorscale='Greys', showscale=False, opacity=alpha))
    fig.data = fig.data[::-1]

    fig.update_layout(
        title='SOM Visualization with Plotly',
        xaxis=dict(title='SOM X Dimension'),
        yaxis=dict(title='SOM Y Dimension', autorange='reversed'),  # Reverse Y axis to match matplotlib's layout
        legend_title_text='Label',
        width=800, height=600
    )
        
    return fig