from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch


tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('./naohai-2b', trust_remote_code=True)
naohai_config = GenerationConfig.from_pretrained('./naohai-2b')
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained('./naohai-2b', trust_remote_code=True, torch_dtype=torch.bfloat16)
model.save_pretrained('./naohai-2b-safetensors')
model.cuda()
all_cuda_0 = all([p.device == torch.device('cuda:0') for p in model.parameters()])
print(f'{all_cuda_0=}')
model.generation_config.max_new_tokens = 100
print(model.generation_config.use_cache)
message = [
    {
        "role": "system",
        "content": "<s>You are NaoHai. A smart and helpful assistant."
    },
    {
        "role": "user",
        "content": "你好, 请问你是由百川智能构建的模型吗？"
    },
    {
        "role": "assistant",
        "content": "对的，我是百川架构的模型.</s>"
    },
    {
        "role": "user",
        "content": "你能给我说一下为什么自注意力机制用的是多头的吗？"
    }
]
output = model.chat(tokenizer, message)
print(output)