import plotly.express as px
import pandas as pd
import numpy as np

def scatter(points, labels=None, label_map=None, color_map=None, size = 1.5, title=None):
    """
    Render an interactive 3D scatter plot using Plotly.

    Args:
        points (np.ndarray): Array of shape (n_points, 3).
        labels (np.ndarray, optional): Array of labels for color grouping.
        label_map (dict, optional): Maps label values to display names.
        size (int, optional): Size of the points.
        title (str, optional): Title of the plot.
    """

    
    if points.shape[1] != 3:
        raise ValueError("points must be of shape (n_points, 3)")

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(points, columns=["x", "y", "z"])

    if labels is not None:
        df["label"] = labels
        if label_map:
            df["label"] = df["label"].map(label_map).fillna(df["label"])
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="label", color_discrete_sequence=color_map)
            
        else:
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="label", color_continuous_scale=color_map, range_color=[0, 1])
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z")

    fig.update_traces(marker=dict(size=size))
    return fig


def combine_plots(figs, rows=1, cols=2):
    """
    Combine multiple plots into a single figure.

    Args:
        figs (list): List of Plotly figures to combine.
        titles (list): List of titles for each plot.
    """
    from plotly.subplots import make_subplots

    combined_fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"}, {"type": "scene"}]]
    )

    for i, fig in enumerate(figs):
        for trace in fig.data:
            combined_fig.add_trace(trace, row = (i // cols) + 1, col = (i % cols) + 1)

    
    combined_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return combined_fig