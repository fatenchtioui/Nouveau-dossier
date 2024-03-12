import tensorflow as tf
from train import * 
# Définir le modèle DNN
model_dnn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(len(feature_columns),)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compiler le modèle
model_dnn.compile(optimizer='adam', loss='mean_squared_error')
# Entraîner le modèle DNN
model_dnn.fit(train[feature_columns], train[target_column], epochs=100, batch_size=32, verbose=1)
