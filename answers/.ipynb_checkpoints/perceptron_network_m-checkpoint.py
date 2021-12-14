# +
epochs = 10000
lr =0.1
for _ in range(epochs):
    # Forward pass. 
    z_hidden = X @ hidden_weights
    hidden_layer = sigmoid(z_hidden)

    z_output = hidden_layer @ output_weights
    y_output = sigmoid(z_output)
    
    # Backpropagation / error calculation
    error_output = y - y_output
    delta_output = error_output * y_output *(1- y_output )
    
    error_hidden = delta_output @ output_weights.T
    delta_hidden = error_hidden * hidden_layer*(1-hidden_layer)
    
    # Update weights. 
    output_weights += lr*hidden_layer.T @ delta_output
    hidden_weights += lr*X.T @ delta_hidden 

print(y)
y_output
