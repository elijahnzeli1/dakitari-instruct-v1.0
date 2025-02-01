from transformers import PreTrainedModel
from .configuration_dakitari_instruct import DakitariInstructConfig

class DakitariInstructPreTrainedModel(PreTrainedModel):
    config_class = DakitariInstructConfig
    base_model_prefix = "dakitari_instruct"
    
    def __init__(self, config):
        super().__init__(config)