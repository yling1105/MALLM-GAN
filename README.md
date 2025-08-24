# MALLM-GAN: Multi-Agent Large Language Model as Generative Adversarial Network for Synthesizing Tabular Data

- Main code: `model_glm.py`
- Example usage on Adult dataset: `adult-glm.ipynb`

## Usage instruction
The implementation of the model is using the HIPAA-compliant Azure GPT service as the backbone LLM. As illustrated in `adult-glm.ipynb`, one need to update the following variables with their own: "api_version", "api_key1", "api_key2", "gen_model_nm", "opt_model_nm".

```
api_version = "2023-05-15"

api_key1 = 777# Your api_key
azure_endpoint1 = "xxx.com/" # Your end point

api_key2 = 777 # Your api_key
azure_endpoint2 = "xxx.com/" # Your end point

gen_client = AzureOpenAI(
            api_key = api_key1,
            api_version = api_version,
            azure_endpoint = azure_endpoint1
        )

opt_client = AzureOpenAI(
    api_key = api_key2,
    api_version = api_version, 
    azure_endpoint = azure_endpoint2
)

gen_model_nm = 'generator4'

opt_model_nm = 'yaobin_gpt4'
```

To use open source model, e.g. models from huggingface, one need to update the generation method in the source code.
