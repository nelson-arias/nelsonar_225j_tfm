import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_backbone(input_tensor):
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((4, 1))(x)
    
    return x

def residual_block(input_tensor, filters, stride=1):
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    shortcut = input_tensor
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet_baseline(input_shape=(64, 256, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=2)
    
    x = layers.GlobalAveragePooling2D(name="global_pooling")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="ResNet_Baseline")

def build_crnn_baseline(input_shape=(64, 256, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape, name="image_input")
    cnn_out = build_cnn_backbone(inputs)
    shape = tf.keras.backend.int_shape(cnn_out)
    sequence_length = shape[2]
    features_dim = shape[3]
    
    sequence = layers.Reshape((sequence_length, features_dim), name="map_to_sequence")(cnn_out)
    
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(sequence)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=False))(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="CRNN_BiLSTM_Baseline")

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        
        self.dense_proj = models.Sequential([
             layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim)
        ])
        
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, training=False, mask=None):
        attention_output = self.attention(inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

def build_hybrid_model(input_shape=(64, 256, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    cnn_out = build_cnn_backbone(inputs)
    
    shape = tf.keras.backend.int_shape(cnn_out)
    sequence_length = shape[2]
    features_dim = shape[3]
    
    sequence = layers.Reshape((sequence_length, features_dim), name="map_to_sequence")(cnn_out)
    
    sequence = PositionalEmbedding(sequence_length=sequence_length, output_dim=features_dim)(sequence)
    
    x = TransformerEncoder(embed_dim=features_dim, dense_dim=512, num_heads=4)(sequence)
    x = layers.Dropout(0.5)(x)
    x = TransformerEncoder(embed_dim=features_dim, dense_dim=512, num_heads=4)(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.GlobalAveragePooling1D(name="author_embedding")(x)
    
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_Transformer")
    return model

def build_model(model_type="hybrid_transformer", input_shape=(64, 256, 1), num_classes=10):
    if model_type == "resnet_baseline":
        return build_resnet_baseline(input_shape, num_classes)
    elif model_type == "crnn_bilstm":
        return build_crnn_baseline(input_shape, num_classes)
    elif model_type == "hybrid_transformer":
        return build_hybrid_model(input_shape, num_classes)
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {model_type}. Elige entre 'resnet_baseline', 'crnn_bilstm', 'hybrid_transformer'.")

if __name__ == "__main__":
    print("SANITY CHECK DE ARQ:")
    dummy_batch = tf.random.normal((4, 64, 256, 1))
    
    for m_type in ["resnet_baseline", "crnn_bilstm", "hybrid_transformer"]:
        m = build_model(m_type)
        preds = m(dummy_batch, training=False)
        params = m.count_params()
        print(f"\nModelo: {m.name}")
        print(f"Parámetros totales: {params:,}")
        print(f"Dimensión salida: {preds.shape}")
        assert preds.shape == (4, 10), f"Error en dimensiones para {m_type}"
        print("Sanity check Superado.")
