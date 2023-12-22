import matplotlib.pyplot as plt
import mplcursors
import numpy as np
# import scann
import umap
import altair as alt

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def plot_heatmap(data, x_labels=None, y_labels=None, title=None):
    fig, ax = plt.subplots(figsize=(50, 3))
    heatmap = ax.pcolor(data, cmap='coolwarm', edgecolors='k', linewidths=0.1)

    # Add color bar to the right of the heatmap
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.remove()

    # Set labels for each axis
    if x_labels:
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if y_labels:
        ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
        ax.set_yticklabels(y_labels, va="center")

    # Set title
    if title:
        ax.set_title(title)

    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_2D(x_values, y_values, labels):

    # Create scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values,
                         y_values,
                         alpha = 0.5,
                         edgecolors='k',
                         s = 40)

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    #aes
    ax.set_title('Embedding visualization in 2D')  # Add a title
    ax.set_xlabel('X_1')  # Add x-axis label
    ax.set_ylabel('X_2')  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels[sel.index])
        sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.5) # Set annotation's background color
        sel.annotation.set_fontsize(12)

    plt.show()


def clusters_2D(x_values, y_values, labels, kmeans_labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values,
                         y_values,
                         c = kmeans_labels,
                         cmap='Set1',
                         alpha=0.5,
                         edgecolors='k',
                         s = 40)  # Change the denominator as per n_clusters

    # Create a mplcursors object to manage the data point interaction
    cursor = mplcursors.cursor(scatter, hover=True)

    #axes
    ax.set_title('Embedding clusters visualization in 2D')  # Add a title
    ax.set_xlabel('X_1')  # Add x-axis label
    ax.set_ylabel('X_2')  # Add y-axis label

    # Define how each annotation should look
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels.category[sel.target.index])
        sel.annotation.get_bbox_patch().set(facecolor='white', alpha=0.95) # Set annotation's background color
        sel.annotation.set_fontsize(14)

    plt.show()

# -----------------------------------------------------------------
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README
# ------------------------------------------------------------------
# def create_index(embedded_dataset,
#                  num_leaves,
#                  num_leaves_to_search,
#                  training_sample_size):

#     # normalize data to use cosine sim as explained in the paper
#     normalized_dataset = embedded_dataset / np.linalg.norm(embedded_dataset, axis=1)[:, np.newaxis]

#     searcher = (
#         scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
#         .tree(
#             num_leaves = num_leaves,
#             num_leaves_to_search = num_leaves_to_search,
#             training_sample_size = training_sample_size,
#         )
#         .score_ah(2, anisotropic_quantization_threshold = 0.2)
#         .reorder(100)
#         .build()
#     )
#     return searcher

#---------------------------------------------------------------
#------------------Embedding search functions------------------------


import numpy as np
import networkx as nx
from random import random, randint
from math import floor, log
np.random.seed(44)

def nearest_neigbor(vec_pos,query_vec):
    nearest_neighbor_index = -1
    nearest_dist = float('inf')

    nodes = []
    edges = []
    for i in range(np.shape(vec_pos)[0]):
        nodes.append((i,{"pos": vec_pos[i,:]}))
        if i<np.shape(vec_pos)[0]-1:
            edges.append((i,i+1))
        else:
            edges.append((i,0))

        dist = np.linalg.norm(query_vec-vec_pos[i])
        if dist < nearest_dist:
            nearest_neighbor_index = i
            nearest_dist = dist

    G_lin = nx.Graph()
    G_lin.add_nodes_from(nodes)
    G_lin.add_edges_from(edges)

    nodes = []
    nodes.append(("*",{"pos": vec_pos[nearest_neighbor_index,:]}))
    G_best = nx.Graph()
    G_best.add_nodes_from(nodes)
    return G_lin, G_best



def layer_num(max_layers: int):
    # new element's topmost layer: notice the normalization by mL
    mL = 1.5
    layer_i = floor(-1 * log(random()) * mL)
    # ensure we don't exceed our allocated layers.
    layer_i = min(layer_i, max_layers-1)
    return layer_i
    #return randint(0,max_layers-1)




def construct_HNSW(vec_pos,m_nearest_neighbor):
    max_layers = 4

    vec_num = np.shape(vec_pos)[0]
    dist_mat = np.zeros((vec_num,vec_num))

    for i in range(vec_num):
        for j in range(i,vec_num):
            dist = np.linalg.norm(vec_pos[i,:]-vec_pos[j,:])
            dist_mat[i,j] = dist
            dist_mat[j,i] = dist

    node_layer = []
    for i in range(np.shape(vec_pos)[0]):
        node_layer.append(layer_num(max_layers))

    max_num_of_layers = max(node_layer) + 1 ## layer indices start from 0
    GraphArray = []
    for layer_i in range(max_num_of_layers):
        nodes = []
        edges = []
        edges_nn = []
        for i in range(np.shape(vec_pos)[0]): ## Number of Vectors
            if node_layer[i] >= layer_i:
                nodes.append((i,{"pos": vec_pos[i,:]}))

        G = nx.Graph()
        G.add_nodes_from(nodes)

        pos=nx.get_node_attributes(G,'pos')

        for i in range (len(G.nodes)):
            node_i = nodes[i][0]
            nearest_edges = -1
            nearest_distances = float('inf')
            candidate_edges = range(0,i)
            candidate_edges_indices = []

            #######################
            for j in candidate_edges:
                node_j = nodes[j][0]
                candidate_edges_indices.append(node_j)

            dist_from_node = dist_mat[node_i,candidate_edges_indices]
            num_nearest_neighbor = min(m_nearest_neighbor,i) ### Add note comment

            if num_nearest_neighbor > 0:
                indices = np.argsort(dist_from_node)
                for nn_i in range(num_nearest_neighbor):
                        edges_nn.append((node_i,candidate_edges_indices[indices[nn_i]]))

            for j in candidate_edges:
                node_j = nodes[j][0]
                dist = np.linalg.norm(pos[node_i]-pos[node_j])
                if dist < nearest_distances:
                    nearest_edges = node_j
                    nearest_distances = dist

            if nearest_edges != -1:
                edges.append((node_i,nearest_edges))

        G.add_edges_from(edges_nn)

        GraphArray.append(G)

    return GraphArray


## Search the Graph
def search_HNSW(GraphArray,G_query):
    max_layers = len(GraphArray)
    G_top_layer = GraphArray[max_layers - 1]
    num_nodes = G_top_layer.number_of_nodes()
    entry_node_r = randint(0,num_nodes-1)
    nodes_list = list(G_top_layer.nodes)
    entry_node_index = nodes_list[entry_node_r]
    #entry_node_index = 26

    SearchPathGraphArray = []
    EntryGraphArray = []
    for l_i in range(max_layers):
        layer_i = max_layers - l_i - 1
        G_layer = GraphArray[layer_i]

        G_entry = nx.Graph()
        nodes = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes.append((entry_node_index,{"pos": p}))
        G_entry.add_nodes_from(nodes)

        nearest_node_layer = entry_node_index
        nearest_distance_layer = np.linalg.norm( G_layer.nodes[entry_node_index]['pos'] - G_query.nodes['Q']['pos'])
        current_node_index = entry_node_index

        G_path_layer = nx.Graph()
        nodes_path = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes_path.append((entry_node_index,{"pos": p}))

        cond = True
        while cond:
            nearest_node_current = -1
            nearest_distance_current = float('inf')
            for neihbor_i in G_layer.neighbors(current_node_index):
                vec1 = G_layer.nodes[neihbor_i]['pos']
                vec2 = G_query.nodes['Q']['pos']
                dist = np.linalg.norm( vec1 - vec2)
                if dist < nearest_distance_current:
                    nearest_node_current = neihbor_i
                    nearest_distance_current = dist

            if nearest_distance_current < nearest_distance_layer:
                nearest_node_layer = nearest_node_current
                nearest_distance_layer = nearest_distance_current
                nodes_path.append((nearest_node_current,{"pos": G_layer.nodes[nearest_node_current]['pos']}))
            else:
                cond = False

        entry_node_index = nearest_node_layer

        G_path_layer.add_nodes_from(nodes_path)
        SearchPathGraphArray.append(G_path_layer)
        EntryGraphArray.append(G_entry)

    SearchPathGraphArray.reverse()
    EntryGraphArray.reverse()

    return SearchPathGraphArray, EntryGraphArray


#
def umap_plot(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)

    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=cols
        #tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart

def umap_plot_big(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=100)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)

    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=cols
        #tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart

def umap_plot_old(sentences, emb):
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    #df_explore = pd.DataFrame(data={'text': qa['text']})
    #print(df_explore)

    #df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = sentences
    df_explore['x'] = umap_embeds[:,0]
    df_explore['y'] = umap_embeds[:,1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=#'x',
        alt.X('x',
            scale=alt.Scale(zero=False)
        ),
        y=
        alt.Y('y',
            scale=alt.Scale(zero=False)
        ),
        tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart
