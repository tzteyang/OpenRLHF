from transformers import AutoModel, AutoConfig

_CONFIG_MAPPING = {
    "qwen2_rm": None
}

def _register_qwen2_rm():
    # adaption for skywork-o1-qwen-prm
    from .configuration_qwen2_rm import Qwen2RMConfig
    from .modeling_qwen2_rm import Qwen2ForRewardModel
    
    _CONFIG_MAPPING["qwen2_rm"] = Qwen2RMConfig
    
    AutoConfig.register("qwen2_rm", Qwen2RMConfig)
    AutoModel.register(Qwen2RMConfig, Qwen2ForRewardModel)

_register_qwen2_rm()

def get_registered_base_class(model_type):
    _registered_model_mapping = {
        "qwen2_rm": AutoModel._model_mapping[_CONFIG_MAPPING["qwen2_rm"]]
    }

    return _registered_model_mapping[model_type]