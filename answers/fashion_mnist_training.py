batch_size=128
step_size_train = X_train.shape[0]// batch_size
epochs=5
model.fit(train_datagen.flow(X_train, y_train_onehot, batch_size=128, shuffle=True),
          validation_data = val_datagen.flow(X_val, y_val_onehot, batch_size=128, shuffle=False),
          steps_per_epoch=step_size_train,
          epochs=epochs)