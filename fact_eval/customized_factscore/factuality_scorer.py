import json
import os
"""
Factuality Scorer Module

This module provides classes for scoring the factuality of language model responses.
It supports different strategies for decomposition, verification, and aggregation of factuality scores.
"""

class FactualityScorer:
    """Base class for factuality scoring.
    
    This class defines the interface for factuality scoring. Subclasses should implement
    the get_score method to provide specific scoring strategies.
    """
    def __init__(self, *args, **kwargs):
        pass

    def get_score(self, prompt_list, response_list):
        """Get factuality scores for a batch of prompts and responses.
        
        Args:
            prompt_list (List[str]): List of prompts
            response_list (List[str]): List of corresponding responses
            
        Returns:
            List[float]: List of factuality scores between 0 and 1
        """
        raise NotImplementedError

class ClaimWiseFactualityScorer(FactualityScorer):
    """Efficient implementation of factuality scoring with configurable strategies.
    
    This scorer supports different strategies for:
    1. Verification Unit: What level to verify (claims or entire response)
    2. Decomposition: How to break down responses into verifiable units
    3. Verification: How to verify each unit against knowledge sources
    4. Aggregation: How to combine individual scores into a final score
    
    Args:
        model_name (str): Base model name for API configuration
        api_base (str, optional): API base URL for the language model
        
        # Decomposition Configuration
        decomposition_model_name (str, optional): Model to use for claim extraction. Examples:
            - "gpt-4.1"
            - "llama3.1-70b-instruction"
        decomposition_prompt (str, optional): Prompt template for claim extraction. One of:
            - "response-to-claims": Extract claims from entire response
            - "sentence-to-claims": Extract claims sentence by sentence
            
        # Search Configuration
        search_framework (str): Framework to use for knowledge retrieval. One of:
            - "google": Use Google Search API
            - "wikipedia": Use Wikipedia API
            - "documents": Use provided documents
        search_model_name (str): Model to use for search/reranking. One of:
            - "bm25-only": Use only BM25 for retrieval
            - "bm25-reranker": Use BM25 + neural reranker
        search_chunk_size (int): Chunk size for document retrieval.
            
        # Verification Configuration
        verification_model_name (str): Model to use for verification. Examples:
            - "gpt-4.1"
            - "llama3.1-70b-instruction"
        verification_prompt (str): Prompt template for verification. Currently only:
            - "default": Standard verification prompt
            
        Aggregation Configuration
            metric (str): The method used to combine individual claim scores into an overall score. Options include:
                - precision: Number of correct claims divided by the total number of claims (#correct_claims / #claims).
                - correctness: Returns 1 if all claims are correct; otherwise, returns 0 (strict all-or-nothing evaluation).
                - F1@K: F1 score for the top K claims, where recall is (#correct_claims / K) and precision is (#correct_claims / K).

    """
    def __init__(
        self,
        # model_name,
        # api_base=None,
        decomposition_model_name=None,
        decomposition_prompt="response-to-claims",
        search_framework="documents",
        search_model_name="bm25-only",
        search_chunk_size=100,
        search_num_chunks=10,
        search_num_processes=1,
        verification_model_name=None,
        verification_prompt="default",
        metric="precision",
        cache_dir=None,
        load_extractor_cache=False,
        *args,
        **kwargs
    ):

        self.decomposition_model_name = decomposition_model_name
        self.decomposition_prompt = decomposition_prompt
        self.search_framework = search_framework
        self.search_model_name = search_model_name
        self.search_chunk_size = search_chunk_size
        self.search_num_chunks = search_num_chunks
        self.search_num_processes = search_num_processes
        self.verification_model_name = verification_model_name
        self.verification_prompt = verification_prompt
        self.metric = metric
        self.load_extractor_cache = load_extractor_cache
        self._cache_dir = cache_dir

        from open_instruct.fact_utils.verifier import OpenaiClaimVerifier
        from open_instruct.fact_utils.extractor import OpenaiClaimExtractor
        from open_instruct.fact_utils.search import SearchEngineDocumentCollection

        self.extractor = OpenaiClaimExtractor(model_name=self.decomposition_model_name)
        self.search_engine = SearchEngineDocumentCollection(chunk_size=self.search_chunk_size)
        self.verifier = OpenaiClaimVerifier(model_name=self.verification_model_name)


    def get_score(self, prompt_list, response_list, docs_list=None, return_metadata=False):
        """Score the factuality of responses.
        
        Args:
            prompt_list (List[str]): List of prompts
            response_list (List[str]): List of corresponding responses
            docs_list (List[List[str]], optional): List of supporting documents for each response
            return_metadata (bool, optional): Whether to return detailed metadata
            
        Returns:
            Union[List[float], Tuple[List[float], Dict]]: Factuality scores and optionally metadata
        """
        
        # save content in a dir named with timestamp
        if self._cache_dir is not None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cache_dir = os.path.join(self._cache_dir, timestamp)
            os.makedirs(self.cache_dir, exist_ok=True)

        if docs_list is None:
            raise ValueError("docs_list is required")

        # # Step 1: Extract claims if verifying at claim level

        if self.load_extractor_cache and self.cache_dir and os.path.exists(os.path.join(self.cache_dir, "extractor_cache.json")):
            with open(os.path.join(self.cache_dir, "extractor_cache.json"), "r") as f:
                prompt_response_to_claims = json.load(f)
            claims_list = [prompt_response_to_claims[prompt_response] for prompt_response in prompt_response_to_claims]
            assert len(claims_list) == len(prompt_list) == len(response_list)
        else:
            claims_list = self.extractor.extract(prompt_list, response_list)
            
            # cache the prompt-response-to-claims
            if self.cache_dir is not None:
                prompt_response_to_claims = {}
                for prompt, response, claims in zip(prompt_list, response_list, claims_list):
                    prompt_response_message = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                    prompt_response_message_str = json.dumps(prompt_response_message)
                    prompt_response_to_claims[prompt_response_message_str] = claims
                extractor_cache_file = os.path.join(self.cache_dir, "extractor_cache.json")
                with open(extractor_cache_file, "w") as f:
                    f.write(json.dumps(prompt_response_to_claims, indent=2))

        all_claims = []
        claim_to_docs = {}
        for prompt, response, claims, docs in zip(prompt_list, response_list, claims_list, docs_list):
            all_claims.extend(claims)
            for claim in claims:
                claim_to_docs[claim] = docs

        # Step 2: Retrieve related documents
        if self.search_framework == "google":
            raise NotImplementedError("Google search is not implemented yet")
        elif self.search_framework == "wikipedia":
            raise NotImplementedError("Wikipedia search is not implemented yet")
        elif self.search_framework == "documents":
            chunks_list = self.search_engine.search(
                documents_list=[claim_to_docs[claim] for claim in all_claims],
                query_list=all_claims,
                k=self.search_num_chunks,
                num_processes=self.search_num_processes,
            )
        else:
            raise ValueError(f"Invalid search framework: {self.search_framework}")
        claim_to_chunks = {claim: chunks for claim, chunks in zip(all_claims, chunks_list)}

        # cache the claim-to-chunks
        if self.cache_dir is not None:
            search_cache_file = os.path.join(self.cache_dir, "search_cache.json")
            with open(search_cache_file, "w") as f:
                f.write(json.dumps(claim_to_chunks, indent=2))

        # Step 3: Verify claims
        correctness_list = self.verifier.verify(
            claim_list=all_claims,
            passages_list=[claim_to_chunks[claim] for claim in all_claims],
        )
        claim_to_correctness = {claim: correctness for claim, correctness in zip(all_claims, correctness_list)}

        # cache the claim-to-correctness
        if self.cache_dir is not None:
            verifier_cache_file = os.path.join(self.cache_dir, "verifier_cache.json")
            with open(verifier_cache_file, "w") as f:
                f.write(json.dumps(claim_to_correctness, indent=2))

        # verify_start_time = time.time()
        # scores_serialized, verifier_metainfo = self.verifier.verify(
        #     claims_serialized, 
        #     query_list=prompt_serialized, 
        #     docs_list=docs_serialized
        # )
        
        # Step 4: Aggregate scores based on strategy
        factuality_score = []
        for claims in claims_list:
            scores_correct = [claim_to_correctness[claim] for claim in claims]
            num_correct_claims = sum(scores_correct)
            num_claims = len(claims)
            score = num_correct_claims / num_claims if num_claims > 0 else None
            # if self.metric == "precision":
            #     score = num_correct_claims / num_claims if num_claims > 0 else 0.5
            # elif self.metric.startswith("f1@"):
            #     K = int(self.metric.removeprefix("f1@"))
            #     prec = num_correct_claims / num_claims if num_claims > 0 else 0.5
            #     recall =  min(num_claims / K, 1.0)
            #     score = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
            # else:
            #     raise ValueError(f"Invalid metric: {self.metric}")
            # if self.aggregation_strategy == "mean":
            #     score = sum(scores_correct) / len(scores_correct) if len(scores_correct) > 0 else 0.0
            # elif self.aggregation_strategy == "min":
            #     score = min(scores_correct)
            # else:
            #     raise ValueError(f"Invalid aggregation strategy: {self.aggregation_strategy}")
            factuality_score.append(score)
        
        output_dict = {
            "factuality_score": factuality_score,
            "num_correct_claims": [sum(bool(claim_to_correctness[claim]) for claim in claims) for claims in claims_list],
            "num_claims": [len(claims) for claims in claims_list],
        }
        metainfo_dict = {
            "claims_list": claims_list,
            "claim_to_chunks": claim_to_chunks,
            "claim_to_correctness": claim_to_correctness,
            "token_usage_extractor": self.extractor.token_usage,
            "token_usage_verifier": self.verifier.token_usage,
        }
        if return_metadata:
            output_dict.update(metainfo_dict)
        return output_dict



if __name__ == "__main__":
    # Test cases with complex, multi-claim responses and mixed factuality
    prompt_list = [
        "Tell me about Albert Einstein's life and contributions to physics.",
        "Describe the history and impact of the Industrial Revolution.",
    ]
    response_list = [
        "Albert Einstein was born in 1879 in Ulm, Germany. He developed the theory of relativity in 1915 while working as a patent clerk in Switzerland. His famous equation E=mc^2 revolutionized physics. He won the 1921 Nobel Prize in Physics for his work on the photoelectric effect. Einstein played the violin professionally and had 5 children.",
        "The Industrial Revolution began in England in the late 18th century. It was marked by the transition from manual production to machine manufacturing, particularly in textiles. Steam power and coal mining were crucial innovations. The average life expectancy doubled during this period, and literacy rates increased to 90%. The first steam engine was invented by Thomas Edison in 1765.",
    ]
    docs_list = [
        [
            "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He published his general theory of relativity in 1915. The famous equation E=mc^2 was introduced in his 1905 paper. Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect. He played violin as a hobby and had three children.",
            "Einstein joined the Institute for Advanced Study at Princeton in 1933 and remained there until his death in 1955. During World War II, he signed a letter to President Roosevelt alerting him to the possibility of Germany developing an atomic bomb, which influenced the creation of the Manhattan Project.",
            "After Einstein's death in 1955, his brain was removed and preserved by pathologist Thomas Harvey for scientific study. Later research showed his brain had an unusually high number of glial cells. Einstein passed the entrance exams and was admitted to ETH Zurich in 1896 on his first attempt."
        ],
        [
            "The Industrial Revolution began in Great Britain in the late 18th century. It marked the transition from hand production to machine manufacturing, starting with the textile industry. Steam power and coal were essential to industrial growth. Life expectancy improved but remained under 40 years in most places. James Watt improved the steam engine design in 1769.",
            "London's population grew dramatically during the Industrial Revolution, increasing from around 1 million in 1800 to 6.7 million in 1900. The invention of the spinning jenny by James Hargreaves in 1764 revolutionized textile production, allowing workers to produce eight threads simultaneously.",
            "Karl Marx published Das Kapital in 1867, providing a critical analysis of capitalism and industrialization. The Stockton and Darlington Railway, opened in 1825, was the world's first public railway to use steam locomotives. Factory working conditions were often harsh, with children working long hours, though the Factory Act of 1833 placed restrictions on child labor."
        ]
    ]
    model_name = "gpt-4.1-mini-standard"
    scorer = ClaimWiseFactualityScorer(
        decomposition_model_name=model_name,
        verification_model_name=model_name,
        search_num_processes=8,
        cache_dir="./cache",
    )
    factuality_score, metainfo = scorer.get_score(prompt_list, response_list, docs_list, return_metadata=True)
    print(factuality_score)
    print(metainfo)

