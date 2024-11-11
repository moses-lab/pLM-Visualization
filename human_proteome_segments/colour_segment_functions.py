
# Ami Sangster




import numpy as np
import pandas as pd
import plotly.graph_objects as go


# only required for work on colab
import os




def sigmoid_RGB_converter(v): 
    """
    convert one dimension of the embedding to a RGB values 
    use a sigmoid function to shift values towards a more uniform distribution for more saturated colours

    Parameters:
    v: input vector of length n

    Returns:
    v_rgb: output vector of length n, 
            values are 2 character strings that represent hexidecimal values spanning from 0-255
    """
    v_centered = v - np.mean(v)
    v_sig = 1/(1 + np.exp(-v_centered))
    v_rgb = [("0"+hex(int(255*x))[2:])[-2:] for x in v_sig]
    return v_rgb


def umap_to_rgb_sigmoid(emb):
    """
    converts a 3D embedding to RGB colours

    Parameters:
    emb: pandas data frame with the 3 dimensions of the embedding in 3 columns with names "X", "Y", and "Z"
            rows correspond to n per-segment embeddings generated from Unsupervised Protein Segmentation

    Returns:
    rgb: a list of n 7 character strings, where each string is a hexidecimal colour that represents
            the high dimsensional space of the per-segment embeddings
            this list is ordered by the rows in the emb[i,:] maps to rbg[i]
    """
    red = sigmoid_RGB_converter(emb["X"])
    green = sigmoid_RGB_converter(emb["Y"])
    blue = sigmoid_RGB_converter(emb["Z"])

    rgb = ["#"+red[ii]+green[ii]+blue[ii] for ii in range(len(red))]
    return rgb



def make_protein_segments_figure(pids, bar_names=None, height=500, width=1000, data_path=None):
    """
    generates a html plotly figure where proteins are shown as horizontal barplots 
    each protein is shown as segments defined by unsupervised protein segmentation
    each segment is coloured to reflect the high dimensional space of the per-segment representation (dim=1024)
    the colour of each segment was generated using a umap of the whole human proteome to reduce the dim to 3
    those 3 dimensions are then converted to RGB (red, green, and blue) hexidecimal values

    Parameters:
    pids: list of UniProt protein IDs
    bar_names: list of names to use in the bar plot,
            if None, pids will be used for names on the bar plot
            otherwise, each name must correspond to a pid (|pids|=|bar_names|)
    height: int for the height of the bar plot
    width: int for the width of the bar plot
    data_path: string of path to data file containing pre-calculated umap coordinates for colours

    Returns:
    fig: html plotly barplot 
    """


    # check inputs
    # if pids is not a list of pids
    if isinstance(pids, str):
        pids = [pids]
        print("Warning, protein ids were not given as a list. \nIf you plan to include multiple protein ids, please format like: pids = ['pid1','pid2',...,'pidN']")
    # if there are no names for the bar chart use the pids
    if bar_names==None:
        bar_names=pids
    # check that there are an equal number of pids and names
    if len(pids) != len(bar_names):
        print("Please enter an equal number of UniProt protein IDs (pids) and names given for the bar chart (bar_names).\nWhere pids[i] corresponds to bar_names[i].")
        return 0
    if data_path == None:
        home_path = os.getenv("HOME")
        # if you are working on colab
        if home_path=="/root":
            data_path = '/content/pLM-Visualization/human_proteome_segments/human_protein_segments_umap_3D.tsv'
        #if the data file is in the same directory as this file and the notebook
        else:
            if "human_protein_segments_umap_3D.tsv" in os.listdir("./"):
                data_path = "human_protein_segments_umap_3D.tsv"
            else:
                print("Data file not found: Please put the data file into the same directory as this notebook")




    # read in a pre-computed umap of the embedding (1024 was umap to 3D, which will be converted to RGB)
    data = pd.read_csv(data_path, sep="\t")
    data.columns = ["X", "Y", "Z", "KEY"]

    # convert the keys to PID and position, then convert data to RGB
    pos = []
    id = []
    for x in data["KEY"]:
        p=x.split(" ")[1].split("-")
        pos.append([int(p[0]), int(p[1])])
        id.append(x.split(" ")[0])
    data["ID"] = id
    data["POS"] = pos
    data["RGB"] = umap_to_rgb_sigmoid(data)

    # check whether given pids are included in this data
    for p in pids:
        if not p in id:
            print("Unfortunately,", p, "is not included in our dataset.\nCommon issues: the protein is too large and excluded from our analysis or not part of the human proteome. \nPlease exlcude this from your analysis and run again.")
            return 0

    fig = go.Figure()

    for ii in range(len(pids)):
        this_pid = pids[ii]
        data_this = data[data["ID"]==this_pid].sort_values("POS").reset_index(drop=True)

        y_name = bar_names[ii]

        # initiate the stacked bar chart
        fig.add_trace(go.Bar(
            y=[y_name],
            x=[0],
            orientation='h',
            marker=dict(
                color="rgb(255,255,255)"),
            name=""))

        # add each domain to the stacked bar chart
        for ii in range(len(data_this)):
            size = data_this["POS"][ii][1]-data_this["POS"][ii][0]
            this_colour=data_this["RGB"][ii]
            fig.add_trace(go.Bar(
                y=[y_name],
                x=[size],
                orientation='h',
                marker=dict(
                    color=this_colour
                ),
                name="["+str(data_this["POS"][ii][0])+", "+str(data_this["POS"][ii][1])+"], rgb("+str(int(this_colour[1:3],16))+
                        ","+str(int(this_colour[3:5],16))+ ","+str(int(this_colour[5:],16))+ ")"
            ))

        # add empty space at the end
        fig.add_trace(go.Bar(
            y=[y_name],
            x=[0],
            orientation='h',
            marker=dict(
                color="rgb(255,255,255)"),
            name=""))
        
        # resize the figure
        fig.update_layout(barmode='stack', plot_bgcolor='rgb(255,255,255)',
            width=width,
            height=height,
            showlegend=False,
            hoverlabel_namelength=-1)

    return fig




