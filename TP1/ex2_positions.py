from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

def pca_plot(position_embeddings, n_positions: int, out_html: str, title: str):
    positions = position_embeddings[:n_positions].detach().cpu().numpy()  

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(positions)

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        text=[str(i) for i in range(len(reduced))],
        color=list(range(len(reduced))),
        title=title,
        labels={"x": "PCA 1", "y": "PCA 2"}
    )

    fig.write_html(out_html)

def main():
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    position_embeddings = model.wpe.weight 

    pca_plot(
        position_embeddings=position_embeddings,
        n_positions=50,
        out_html="TP1/positions_50.html",
        title="PCA : positions 0–50"
    )

    pca_plot(
        position_embeddings=position_embeddings,
        n_positions=200,
        out_html="TP1/positions_200.html",
        title="PCA, positions 0–200"
    )

if __name__ == "__main__":
    main()