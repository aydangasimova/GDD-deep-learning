# +
epochs = 10000
for _ in range(epochs):
    # Forward pass. 
    hidden_layer = X @ hidden_weights
    hidden_activated = sigmoid(hidden_layer)

    output_layer = hidden_activated @ output_weights
    output_activated = sigmoid(output_layer)
    y_hat = output_activated
    
    # Backpropagation / error calculation
    error_output = y - y_hat
    delta_output = error_output * sigmoid_derivative(output_activated)
    
    error_hidden = delta_output @ output_weights.T
    delta_hidden = error_hidden * sigmoid_derivative(hidden_activated)
    
    # Update weights. 
    output_weights += hidden_activated.T @ delta_output
    hidden_weights += X.T @ delta_hidden 
    
print(y)
y_hat
