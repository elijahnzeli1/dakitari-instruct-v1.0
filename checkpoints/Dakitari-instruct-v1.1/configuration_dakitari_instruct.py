from transformers import PretrainedConfig

class DakitariInstructConfig(PretrainedConfig):
    model_type = "dakitari_instruct"
    
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=6,
                 num_attention_heads=8, intermediate_size=512, max_position_embeddings=512,
                 layer_norm_eps=1e-6, dropout=0.1, initializer_range=0.02, use_cache=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache