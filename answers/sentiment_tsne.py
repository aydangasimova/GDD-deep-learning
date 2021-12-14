from sklearn.manifold import TSNE


embedding_weights = model.layers[0].get_weights()[0]
embedding_2d = TSNE().fit_transform(embedding_weights[:1000, :])
plt.plot(embedding_2d[:, 0], embedding_2d[:, 1], "o")
