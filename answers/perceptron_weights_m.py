
n_input = 2
n_hidden = 3
n_output = 1

hidden_weights = np.random.uniform(size=(n_input, n_hidden))
output_weights = np.random.uniform(size=(n_hidden, n_output))

print(hidden_weights.shape)
print(output_weights.shape)
