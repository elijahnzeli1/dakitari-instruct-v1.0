import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import numpy as np

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")
        
        self.head_dim = embed_dim // num_heads
        
        # Initialize dense layers as None
        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.combine_heads = None
    
    def build(self, input_shape):
        # Create dense layers with proper shapes
        self.query_dense = layers.Dense(self.embed_dim, use_bias=False, 
                                      kernel_initializer='glorot_uniform')
        self.key_dense = layers.Dense(self.embed_dim, use_bias=False, 
                                    kernel_initializer='glorot_uniform')
        self.value_dense = layers.Dense(self.embed_dim, use_bias=False, 
                                      kernel_initializer='glorot_uniform')
        self.combine_heads = layers.Dense(self.embed_dim, 
                                        kernel_initializer='glorot_uniform')
        
        # Build all dense layers with input shape
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        
        # Build combine heads layer with appropriate shape
        combine_input_shape = tf.TensorShape([input_shape[0], input_shape[1], self.embed_dim])
        self.combine_heads.build(combine_input_shape)
        
        super().build(input_shape)
    
    def split_heads(self, x, batch_size):
        """Reshape and transpose to separate heads"""
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def attention(self, query, key, value, mask=None):
        # Convert all inputs to float32
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            
        # Compute attention scores
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale scores
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Compute weighted sum of values
        output = tf.matmul(attention_weights, value)
        
        return output
    
    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # Convert input to float32
        x = tf.cast(x, tf.float32)
        
        # Linear projections
        query = self.query_dense(x)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(x)      # (batch_size, seq_len, embed_dim)
        value = self.value_dense(x)  # (batch_size, seq_len, embed_dim)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention
        scaled_attention = self.attention(query, key, value, mask)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape and combine heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim)
        concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.embed_dim])
        
        # Final linear projection
        output = self.combine_heads(concat_attention)
        return output

class ExpertLayer(layers.Layer):
    def __init__(self, ff_dim, input_dim):
        super().__init__()
        self.ff_dim = ff_dim
        self.input_dim = input_dim
        
        # Initialize layers
        self.dense1 = layers.Dense(ff_dim, activation="gelu")
        self.dense2 = layers.Dense(input_dim)
    
    def build(self, input_shape):
        # Build dense layers
        self.dense1.build(input_shape)
        
        # Build second dense layer with shape from first dense layer's output
        dense2_input_shape = tf.TensorShape([input_shape[0], input_shape[1], self.ff_dim])
        self.dense2.build(dense2_input_shape)
        
        self.built = True
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "ff_dim": self.ff_dim,
            "input_dim": self.input_dim
        })
        return config

class MixtureOfExperts(layers.Layer):
    def __init__(self, input_dim, num_experts, expert_dim, capacity_factor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.capacity_factor = capacity_factor
        
        # Create experts
        self.experts = [
            ExpertLayer(expert_dim, input_dim) 
            for _ in range(num_experts)
        ]
        
        # Router network (gates)
        self.router = layers.Dense(num_experts, use_bias=False)
        
    def build(self, input_shape):
        # Build router network
        router_input_shape = tf.TensorShape([input_shape[0], input_shape[1], self.input_dim])
        self.router.build(router_input_shape)
        
        # Build each expert
        for expert in self.experts:
            expert.build(input_shape)
        
        self.built = True
    
    def call(self, inputs, training=False):
        # Ensure inputs are float16
        inputs = tf.cast(inputs, dtype=tf.float16)  # Cast inputs to float16
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Compute routing weights
        router_logits = self.router(inputs)  # Shape: (batch_size, seq_len, num_experts)
        
        # Apply softmax to get routing probabilities
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts (k=2 in this implementation)
        k = 2
        top_k_gates, top_k_indices = tf.math.top_k(router_probs, k=k)
        
        # Normalize the gates
        top_k_gates = tf.nn.softmax(top_k_gates, axis=-1)
        
        # Initialize expert outputs
        expert_outputs = tf.zeros([batch_size, seq_len, self.input_dim], dtype=inputs.dtype)
        
        # Compute capacity
        capacity = tf.cast(tf.math.ceil(
            self.capacity_factor * tf.cast(batch_size * seq_len, tf.float32) / self.num_experts
        ), tf.int32)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = tf.reduce_any(tf.equal(top_k_indices, i), axis=-1)
            expert_mask = tf.cast(expert_mask, dtype=tf.float16)  # Cast expert_mask to float16
            expert_mask = tf.expand_dims(expert_mask, -1)
            
            # Get expert inputs
            expert_input = inputs * expert_mask
            
            # Apply expert
            expert_output = self.experts[i](expert_input)
            
            # Combine expert outputs weighted by gates
            gate_mask = tf.cast(tf.equal(top_k_indices, i), tf.float16)
            gates = top_k_gates * gate_mask
            gates = tf.reduce_sum(gates, axis=-1, keepdims=True)
            expert_outputs += expert_output * gates
        
        return expert_outputs

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, num_experts=8, expert_capacity=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.expert_capacity = expert_capacity  # Store expert_capacity
        
        # Use expert_capacity in MoE routing
        self.moe = MixtureOfExperts(
            input_dim=embed_dim,
            num_experts=num_experts,
            expert_dim=ff_dim,
            capacity_factor=expert_capacity/embed_dim  # Use expert_capacity here
        )
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def build(self, input_shape):
        # Build attention layer
        self.att.build(input_shape)
        
        # Build MoE layer
        self.moe.build(input_shape)
        
        # Build normalization and dropout layers
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        
        super().build(input_shape)
    
    def call(self, inputs, training=False, mask=None):
        # Self-attention
        attn_output = self.att(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # MoE layer
        moe_output = self.moe(out1, training=training)
        moe_output = self.dropout2(moe_output, training=training)
        return self.layernorm2(out1 + moe_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout1.rate,
            "num_experts": self.moe.num_experts,
            "expert_capacity": self.expert_capacity
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def build(self, input_shape):
        # Build token embedding
        token_input_shape = tf.TensorShape([input_shape[0], input_shape[1]])
        self.token_emb.build(token_input_shape)
        
        # Build position embedding
        pos_input_shape = tf.TensorShape([input_shape[0], input_shape[1]])
        self.pos_emb.build(pos_input_shape)
        
        self.built = True
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

class DakitariInstructModel(keras.Model):
    def __init__(
            self,
            vocab_size=30522,
            maxlen=512,
            embed_dim=1024,
            num_heads=16,
            ff_dim=4096,
            num_transformer_blocks=24,
            dropout_rate=0.1,
            num_experts=8,
            expert_capacity=128,
            **kwargs
        ):
        super().__init__(**kwargs)
        # Store all parameters as instance attributes
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        
        # Initialize layers
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, num_experts, expert_capacity)
            for _ in range(num_transformer_blocks)
        ]
        self.dropout = layers.Dropout(dropout_rate)
        self.final_layer = layers.Dense(vocab_size, activation="softmax")
    
    def build(self, input_shape):
        # Handle different input types
        if isinstance(input_shape, dict):
            input_shape = input_shape['input_ids']
        elif isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        
        # Convert to TensorShape if not already
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        
        # Build embedding layer
        self.embedding_layer.build(input_shape)
        
        # Build transformer blocks with proper shape
        embedding_shape = tf.TensorShape([input_shape[0], input_shape[1], self.embed_dim])
        for block in self.transformer_blocks:
            block.build(embedding_shape)
        
        # Build final layer
        self.final_layer.build(embedding_shape)
        
        self.built = True
    
    def call(self, inputs, training=False):
        # Extract input tensors from the dictionary
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Continue with your model logic using input_ids and attention_mask
        # Example: mask = tf.ones((batch_size, seq_len))
        mask = tf.ones((batch_size, seq_len))

        # Create attention mask if provided
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        
        # Embedding layer
        x = self.embedding_layer(input_ids)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=attention_mask)
        
        # Final processing
        x = self.dropout(x, training=training)
        return self.final_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "dropout_rate": self.dropout_rate,
            "num_experts": self.num_experts,
            "expert_capacity": self.expert_capacity
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)