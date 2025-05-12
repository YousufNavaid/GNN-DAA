import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import dgl
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv, GATConv
from sklearn.manifold import TSNE
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    data = CoraGraphDataset()
    g = data[0]
    return g, data.num_classes

# Define GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, num_classes)

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        return self.conv2(g, x)

# Define GAT model
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads)
        self.gat2 = GATConv(hidden_feats * heads, num_classes, 1)

    def forward(self, g, x):
        x = F.elu(self.gat1(g, x))
        x = x.flatten(1)
        x = self.gat2(g, x)
        return x.mean(1)

# Train model
def train(model, g, features, labels, train_mask, test_mask, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_list, acc_list = [], []

    for epoch in range(epochs):
        model.train()
        out = model(g, features)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, g, features, labels, test_mask)
        loss_list.append(loss.item())
        acc_list.append(acc)

    return loss_list, acc_list

# Evaluate model
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        out = model(g, features)
        preds = out.argmax(1)
        correct = (preds[mask] == labels[mask]).sum().item()
        return correct / mask.sum().item()

# Convert DGL graph to NetworkX for visualization
def plot_graph(g, labels, num_classes=7):
    nx_g = g.to_networkx()
    fig, ax = plt.subplots(figsize=(12, 12))
    
    pos = nx.spring_layout(nx_g, seed=42)
    node_colors = [labels[node].item() for node in range(g.num_nodes())]
    cmap = plt.get_cmap("tab20", num_classes)

    nx.draw(nx_g, pos, node_color=node_colors, cmap=cmap, 
            with_labels=False, node_size=20, edge_color="gray", alpha=0.7, ax=ax)

    ax.set_title("CORA Graph Structure (nodes colored by label)")
    st.pyplot(fig)
    plt.close(fig)


def plot_embeddings(model, g, features, labels):
    model.eval()
    with torch.no_grad():
        embeddings = model(g, features)

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings.detach().cpu().numpy())

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels.cpu().numpy(),
                         cmap='tab10', s=30, alpha=0.8)
    ax.set_title("t-SNE of Node Embeddings (colored by label)")
    st.pyplot(fig)
    plt.close(fig)



# Streamlit UI
st.title("Graph Neural Networks on CORA Dataset")
g, num_classes = load_data()
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
test_mask = g.ndata['test_mask']

# Sidebar options
model_name = st.sidebar.selectbox("Choose Model", ["GCN", "GAT"])
hidden_feats = st.sidebar.slider("Hidden Units", 4, 64, 16)
epochs = st.sidebar.slider("Epochs", 100, 500, 200)

# Visualization of the graph structure
if st.sidebar.checkbox("Show Graph Structure"):
    with st.spinner("Visualizing graph..."):
        plot_graph(g, labels, num_classes)

if st.sidebar.button("Train"):
    with st.spinner("Training... Please wait."):
        if model_name == "GCN":
            model = GCN(features.shape[1], hidden_feats, num_classes)
        else:
            model = GAT(features.shape[1], hidden_feats, num_classes)

        loss_list, acc_list = train(model, g, features, labels, train_mask, test_mask, epochs)
        final_acc = evaluate(model, g, features, labels, test_mask)

        st.success(f"✅ Training completed for {model_name}")
    st.metric("Final Test Accuracy", f"{final_acc:.2%}")

    # Plot loss & accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_list)
    ax1.set_title("Loss over Epochs")
    ax2.plot(acc_list, color='green')
    ax2.set_title("Accuracy over Epochs")
    st.pyplot(fig)

    # ====== t-SNE Visualization (Predicted Labels) ======
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix
    import plotly.express as px
    import pandas as pd
    import seaborn as sns

    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        embeddings = logits.detach().numpy()
        predicted = logits.argmax(1).cpu().numpy()
        true_labels = labels.cpu().numpy()

    # Run t-SNE on final embeddings
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Convert to DataFrame for Plotly
    df = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "True Label": true_labels,
        "Predicted Label": predicted
    })

    st.subheader("t-SNE of Node Embeddings (by **Predicted Label**)")
    fig_tsne = px.scatter(df, x="x", y="y",
                          color=df["Predicted Label"].astype(str),
                          hover_data=["True Label"],
                          title="t-SNE Plot (Predicted Labels)",
                          height=600)
    st.plotly_chart(fig_tsne, use_container_width=True)

    # ====== Confusion Matrix ======
    st.subheader("📉 Confusion Matrix (on Test Set)")
    from sklearn.metrics import ConfusionMatrixDisplay

    test_preds = logits[test_mask].argmax(1).cpu().numpy()
    test_true = labels[test_mask].cpu().numpy()

    cm = confusion_matrix(test_true, test_preds)
    fig_cm, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

