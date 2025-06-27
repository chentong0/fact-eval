import spacy
from vllm import LLM
from transformers import AutoTokenizer
import torch
import gc
from tqdm import tqdm


class ClaimExtractor():
    def __init__(self, model_name, lazy_loading=True):
        self.llm = None
        self.client = None
        self.model_name = model_name
        self.lazy_loading = lazy_loading

        try:
            self.spacy_nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            # install spacy model
            from spacy.cli import download
            download("en_core_web_sm")
            self.spacy_nlp = spacy.load('en_core_web_sm')

        if not self.lazy_loading:
            self.load_model()

    def load_model(self):
        if "gpt" in self.model_name:
            import os
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

    def get_prompt(self, snippet):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

### Input:
{snippet}

### Response:""".strip()

    def get_prompt_gpt(self, snippet, sentence):
        return f"""
You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

Extract *verifiable atomic* facts. Write one fact per line. Do not include any numbering or bullet points.

Text: {snippet}
Sentence to be focused on: {sentence}
Facts:""".strip()

    def batch_scanner_extractor(self, inputs):
        """
        Process multiple inputs (questions and/or responses) in a batch and return extracted claims.

        Args:
            inputs (list of dict): Each dict contains either:
                - {"output": <output>} for non-QA inputs
                - {"question": <question>, "output": <output>} for QA inputs
            is_qa (bool): Whether the inputs are QA-based or not.
            cost_estimate_only (bool): If True, only estimate the cost without extracting claims.

        Returns:
            list of dict: Each dict contains the extracted claims for the corresponding input.
        """
        snippet_list = []
        for input_data in inputs:
            question = input_data["question"].strip()
            response = input_data["output"].strip()
            sentences = self.get_sentence(response)
            for i, sentence in enumerate(sentences):
                # if sentence contains less than five words, skip it
                # if len(sentence.split()) < 5:
                #     continue
                # snippet = f"Questions:\n{question}\nResponse:\n{response.replace(sentence, f'<SOS>{sentence}<EOS>')}"

                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])
                # remove new liens in the context
                snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # remove new liens in the snippet and remove all "##" and "###"
                snippet = snippet.replace("\n", " ")
                snippet = snippet.replace("####", "").replace("###", "").replace("##", "")

                snippet = f"Question: {question.strip()}\nResponse: {snippet}".strip()

                snippet_list.append({"snippet": snippet, "sentence": sentence})

        # Use batch_fact_extractor to process all snippet_list
        batch_results = self.batch_fact_extractor(snippet_list)

        # Group results back into the original input structure
        grouped_results = []
        snippet_idx = 0
        for input_data in inputs:
            response = input_data["output"].strip()

            sentences = self.get_sentence(response)
            claims_per_input = []
            for _ in sentences:
                claims_per_input.append(batch_results[snippet_idx]["claims"])
                snippet_idx += 1

            grouped_results.append({"claims": claims_per_input})

        return grouped_results

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]


    def batch_fact_extractor(self, snippet_list):
        """
        Process multiple snippets in a batch and return a list of dictionaries with extracted claims.

        Args:
            snippet_list (list of dict): List of text snippets to process.

        Returns:
            list of dict: Each dict contains the extracted claims for a snippet.
        """

        if self.lazy_loading:
            self.load_model()

        if self.llm:
            from vllm import SamplingParams
            prompts = []
            results = []

            for snippet in snippet_list:
                # formatted_input = self.alpaca_prompt.format(snippet, "")
                formatted_input = self.get_prompt(snippet["snippet"])
                # prompt = self.get_prompt(snippet)
                formatted_input = self.tokenizer.apply_chat_template([{"role": "user", "content": formatted_input}], tokenize=False, add_generation_prompt=True)
                if self.tokenizer.bos_token and formatted_input.startswith(self.tokenizer.bos_token):
                    formatted_input = formatted_input.removeprefix(self.tokenizer.bos_token)
                
                prompts.append(formatted_input)

            outputs = self.llm.generate(
                prompts, sampling_params=SamplingParams(max_tokens=512, temperature=0, stop="\n\n")
            )

            for i, output in enumerate(outputs):
                clean_output = output.outputs[0].text
                if not clean_output or "No verifiable claim." in clean_output:
                    results.append({"claims": []})
                else:
                    claims = [x.strip() for x in clean_output.split("\n") if len(x.split()) > 1]    # at least two words
                    results.append({"claims": claims})

                # print(f"---\nSnippet: {prompts[i]}\n---")
                # print(f"---\nResponse: {clean_output}\n---")
        else:
            results = []
            outputs = []
            completion_tokens = 0
            prompt_tokens = 0
            MAX_TRIES = 3
            progress_bar = tqdm(total=len(snippet_list), desc="Processing Prompts", unit="prompt")
            for snippet in snippet_list:
                for tries in range(MAX_TRIES):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": self.get_prompt_gpt(snippet["snippet"], snippet["sentence"])}
                            ],
                            max_tokens=512,
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
                    print(f"Failed to generate response for prompt: {snippet['sentence']} after {MAX_TRIES} tries.")
                    outputs.append(None)

            for i, output in enumerate(outputs):
                response = output.choices[0].message.content
                clean_output = response.strip()
                if not clean_output or "No verifiable claim." in clean_output:
                    results.append({"claims": []})
                else:
                    claims = [x.strip() for x in clean_output.split("\n") if len(x.split()) > 1]    # at least two words
                    results.append({"claims": claims})

        if self.lazy_loading:
            self.unload_model()
            
        return results


if __name__ == "__main__":
    # Example usage of ClaimExtractor
    # model_name = "./model/mistral_based_claim_extractor"  # Replace with your model name or path
    model_name = "gpt-4.1-mini-standard"
    
    claim_extractor = ClaimExtractor(model_name=model_name)

    # Test input
    test_inputs = [
        {
            "question": "Tell me about France",
            "output": "France is a country in Western Europe known for its rich history, strong cultural identity, and global influence. It is one of the largest countries in Europe, both in terms of area and population.  France borders Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra. It also has coastlines along the Atlantic Ocean, English Channel, and Mediterranean Sea. Its landscape includes mountains like the Alps and Pyrenees, river systems such as the Seine and the Rhône, and fertile plains.  France is a republic with a semi-presidential system. The President is the head of state, and the Prime Minister is the head of government. France is a founding member of the European Union (EU) and a permanent member of the United Nations Security Council.  France has one of the largest economies in the world. It is diverse, with major industries in aerospace, automotive, luxury goods, agriculture, energy, and tourism. It is also a leader in nuclear energy production.  France is widely regarded as a cultural center. It has contributed significantly to art, philosophy, science, fashion, and cuisine. Paris, its capital, is especially known for its museums (like the Louvre), landmarks (like the Eiffel Tower), and historic neighborhoods.  French is the official language. The country has a centralized education system and is home to some of the world's top universities and research institutions.  France has a long and complex history. It was a major power during the Middle Ages, played a central role in the Enlightenment, and experienced major political changes during events like the French Revolution. It was involved in both World Wars and has since become a stable democratic state.  Let me know if you want information on a specific aspect, such as cuisine, history, or regional differences." 
        },
        {
            "question": "What are the fundamental elements that James and Eleanor Gibson's ecological theory of perception propose, and how does this theory explain the way humans perceive their environment?",
            "output": "The ecological theory of perception, proposed by James J. Gibson and his wife Eleanor Gibson, fundamentally challenges traditional notions of perception as a passive process. This groundbreaking theory posits that perception is an active process, closely linked to an individual's interaction with their environment.\n\nAt its core, the ecological theory of perception consists of three fundamental elements:\n\n1. **Affordances**: This concept, introduced by James Gibson, refers to the possible actions or uses that an object or environment affords an individual. In other words, it describes the potential opportunities or constraints presented by the environment, influencing an individual's perception and behavior.\n2. **Perceptual readiness**: Gibson argued that perception is not a static process, but rather a dynamic one that is influenced by an individual's prior experiences, knowledge, and expectations. This readiness prepares an individual to perceive and interpret their environment.\n3. **Action-perception cycle**: The ecological theory of perception suggests that perception and action are tightly linked. An individual's perception of their environment is influenced by their actions, and their actions are guided by their perception of the environment.\n\nAccording to this theory, humans perceive their environment through a continuous process of exploration and interaction. As we move through our environment, we gather information about the affordances, layout, and structure of the space. This information is then used to guide our actions, such as navigating, reaching, or manipulating objects.\n\nThe ecological theory of perception has far-reaching implications for our understanding of perception, cognition, and human"
        },
        {
            "question": "What are the fundamental elements that James and Eleanor Gibson's ecological theory of perception propose, and how does this theory explain the way humans perceive their environment?",
            "output": "James and Eleanor Gibson's ecological theory of perception, developed in the late 20th century, fundamentally challenges traditional theories of perception by emphasizing the environment itself as the source of sensory information. This theory posits that the environment is a dynamic, three-dimensional space filled with information that organisms can use to interact with their surroundings.\n\n### Fundamental Elements of the Ecological Theory of Perception\n\n1. **Affordances**: Gibson introduced the concept of affordances, which refers to the potential actions an object or environment offers to the organism. For instance, a chair affords sitting, while a railing affords support.\n\n2. **Affordance Space**: This is the collection of all the affordances available in an environment. It represents the entirety of the information available to an organism for action and perception.\n\n3. **Selection and Invariance**: Gibson argues that organisms must select a subset of the available information that is relevant for their immediate needs, a process he calls perceptual selection. This information is invariant, meaning it remains consistent under different viewing conditions, such as changes in lighting or distance.\n\n4. **Direct Perception**: Unlike earlier theories that emphasized the role of intermediary processes like perception or thought, Gibson proposed direct perception. According to this view, organisms perceive directly from the environment, bypassing internal cognitive processes.\n\n5. **Sensory Systems**: The theory focuses on how the sensory systems of organisms interact with their environment. It posits that these systems are adapted to their ecological niches,"
        },
        {
            "question": "Who is Robert Sternberg and what is his Triarchic Theory of Intelligence's applicability in the field of cognitive psychology?", 
            "output": "Robert Sternberg is a prominent psychologist and cognitive theorist known for developing the Triarchic Theory of Intelligence. He serves as a professor at Tufts University and has made significant contributions to the field of cognitive psychology through his innovative theories on intelligence.\n\nSternberg's Triarchic Theory of Intelligence, introduced in the 1980s, posits that intelligence is composed of three distinct facets: analytical, creative, and practical. Each component represents a different mode of intellectual functioning:\n\n1. **Analytical Intelligence**: This aspect of intelligence pertains to logical reasoning, problem-solving, and critical thinking skills. It is closely associated with traditional forms of intelligence testing, such as those based on I.Q. scores, which measure one's ability to reason and solve problems through logic.\n\n2. **Creative Intelligence**: This component involves the capacity to generate new ideas and think flexibly. It encompasses the ability to find novel solutions to problems and to innovate in different fields.\n\n3. **Practical Intelligence**: Often referred to as street-smarts, practical intelligence is the ability to effectively apply knowledge in everyday life. It involves the skill of adapting to new situations and understanding how to use resources effectively in real-world contexts.\n\nThe applicability of Sternberg's Triarchic Theory in the field of cognitive psychology is considerable. It provides a more comprehensive framework for understanding human intelligence beyond the limitations of traditional intelligence tests. This theory can be applied in various areas, including educational psychology, career counseling, and organizational behavior"
        },
    ]

    # Run batch_scanner_extractor
    extracted_claims = claim_extractor.batch_scanner_extractor(test_inputs)

    # Print the results
    for idx, result in enumerate(extracted_claims):
        print(f"Input {idx + 1}:")
        print(f"Claims: {result['claims']}")
        print("-" * 50)
