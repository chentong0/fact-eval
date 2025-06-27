import os
import json

from vllm import LLM
from transformers import AutoTokenizer
import torch
import gc
from tqdm import tqdm

class ClaimVerifier():
    def __init__(self, model_name, lazy_loading=True):
        # self.model = None
        self.llm = None
        self.client = None
        self.model_name = model_name
        self.lazy_loading = lazy_loading

    def apply_chat_template(self, messages):
        system_message = messages[0]["content"]
        user_message = messages[1]["content"]

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def get_message(self, snippet):
        instruction = """You need to judge whether a claim is supported or not by search results from Google. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the two categories:

Supported: A claim is supported by the search results if one or more search results directly support the claim. There can be cases where some search results are not fully related to the claim but no search result should directly contradict the claim. All parts of a claim should be supported by the search results. If there is a part of a claim that is not directly supported, the claim should be marked as unsupported.
Unsupported: If a claim is not supported by the search results, mark it as unsupported.""".strip()
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": snippet}
        ]

    def get_prompt_gpt(self, snippet):
        return f"""
You need to judge whether a claim is supported or not by search results from Google. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Mark your answer with ### signs.

Below are the definitions of the two categories:

Supported: A claim is supported by the search results if one or more search results directly support the claim. There can be cases where some search results are not fully related to the claim but no search result should directly contradict the claim. All parts of a claim should be supported by the search results. If there is a part of a claim that is not directly supported, the claim should be marked as unsupported.
Unsupported: If a claim is not supported by the search results, mark it as unsupported.

{snippet}""".strip()

    def load_model(self):
        if "gpt" in self.model_name:
            import openai
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            )
            print(os.getenv("AZURE_OPENAI_API_VERSION", ""))
            print(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
            print(os.getenv("AZURE_OPENAI_API_KEY", ""))
        else:
            # get the available gpu memory, and the model should take 20gb
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - torch.cuda.memory_allocated()
            # Calculate memory utilization based on available memory vs desired 20GB
            desired_memory = 20 * 1024 * 1024 * 1024  # 20GB in bytes
            gpu_memory_utilization = min(1.0, desired_memory / total_memory)

            self.llm = LLM(
                model=self.model_name, 
                dtype=torch.bfloat16,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=4096,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def unload_model(self):
        if self.llm:
            del self.llm
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        elif self.client:
            del self.client
            gc.collect()

    def batch_verifying_claim(self, claim_snippets_dict, search_res_num=5):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """

        if self.lazy_loading:
            self.load_model()

        prompts = []
        results = {}
        for claim, search_snippet_lst in claim_snippets_dict.items():
            search_res_str = ""
            search_cnt = 1
            for search_dict in search_snippet_lst[:search_res_num]:
                search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1
                
            usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"

            prompts.append(usr_input)

        if self.llm:
            for i, usr_input in enumerate(prompts):
                # prompt = self.alpaca_prompt.format(self.instruction, usr_input)
                # prompt = self.get_prompt(usr_input)
                message = self.get_message(usr_input)
                prompt = self.apply_chat_template(message)

                if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
                    prompt.removeprefix(self.tokenizer.bos_token)
                prompts[i] = prompt

            from vllm import SamplingParams
            outputs = self.llm.generate(
                prompts, sampling_params=SamplingParams(max_tokens=16, temperature=0)
            )
            for i, (claim, output) in enumerate(zip(claim_snippets_dict.keys(), outputs)):
                response = output.outputs[0].text
                clean_output = response.strip()
                results[claim] = clean_output == "supported"
        else:
            outputs = []
            completion_tokens = 0
            prompt_tokens = 0
            MAX_TRIES = 3
            progress_bar = tqdm(total=len(prompts), desc="Processing Prompts", unit="prompt")
            for prompt in prompts:
                for tries in range(MAX_TRIES):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                # {"role": "system", "content": self.instruction},
                                {"role": "user", "content": self.get_prompt_gpt(prompt)}
                            ],
                            max_tokens=16,
                            temperature=0
                        )
                        outputs.append(response)
                        completion_tokens += response.usage.completion_tokens
                        prompt_tokens += response.usage.prompt_tokens
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            "Completion Tokens": completion_tokens,
                            "Prompt Tokens": prompt_tokens,
                        })
                        break
                    except Exception as e:
                        print(f"Error: {e}. Retrying {tries + 1}/{MAX_TRIES}...")
                else:
                    print(f"Failed to generate response for prompt: {prompt} after {MAX_TRIES} tries.")
                    outputs.append(None)

                # print("-" * 20, prompt, "-" * 20)
                # print("-" * 20, response.choices[0].message.content, "-" * 20)
                # print(response)
            for i, (claim, output) in enumerate(zip(claim_snippets_dict.keys(), outputs)):
                if output is not None:
                    response = output.choices[0].message.content
                    clean_output = response.strip()
                    results[claim] = clean_output == "supported"
                else:
                    results[claim] = False
                


        # json.dump({claim: {"prompt": output.prompt, "response": output.outputs[0].text} for claim, output in zip(claim_snippets_dict.keys(), outputs)}, open("debug.json", "w"), indent=4)
        
        if self.lazy_loading:
            self.unload_model()

        return results

if __name__ == "__main__":
    # Example usage of ClaimVerifier
    # model_name = "../../models/llama3_based_claim_verifier"  # Replace with your model name or path
    model_name = "gpt-4.1-mini-standard"

    claim_snippets_dict = {
        "The Eiffel Tower is located in Paris.": [
            {"title": "Eiffel Tower - Wikipedia", "snippet": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "link": "https://en.wikipedia.org/wiki/Eiffel_Tower"},
            {"title": "Eiffel Tower Facts", "snippet": "The Eiffel Tower is one of the most iconic landmarks in Paris, France.", "link": "https://www.toureiffel.paris/en"},
        ],
        "The Eiffel Tower is located in London.": [
            {"title": "Eiffel Tower - Wikipedia", "snippet": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "link": "https://en.wikipedia.org/wiki/Eiffel_Tower"},
            {"title": "Eiffel Tower Facts", "snippet": "The Eiffel Tower is one of the most iconic landmarks in Paris, France.", "link": "https://www.toureiffel.paris/en"},
        ],
        "The sky is pink.": [
            {"title": "Sky Color - Wikipedia", "snippet": "The sky appears blue due to Rayleigh scattering of sunlight.", "link": "https://en.wikipedia.org/wiki/Sky_color"},
            {"title": "Why is the sky blue?", "snippet": "The blue color of the sky is due to the scattering of light by the atmosphere.", "link": "https://www.scientificamerican.com/article/why-is-the-sky-blue/"},
        ],
    }

    # Initialize the ClaimVerifier
    verifier = ClaimVerifier(model_name=model_name)

    # Run the claim verification
    results = verifier.batch_verifying_claim(claim_snippets_dict)

    # Print the results
    print("Verification Results:")
    # for result in results:
    print(json.dumps(results, indent=4))
