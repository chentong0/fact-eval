import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy
from fact_eval.extractor.veriscore_extractor import ClaimExtractor
from fact_eval.search.dense_index_search import LocalSearchAPI
from fact_eval.utils.metrics import get_veriscore
from fact_eval.verifier.veriscore_verifier import ClaimVerifier
from tqdm import tqdm


@dataclass
class VeriScoreConfig:
    model_name_extraction: str = None
    model_name_verification: str = None
    search_passages_path: Optional[str] = None
    search_passages_embedding_path: Optional[str] = None
    search_model_name: Optional[str] = None
    search_res_num: int = 5
    cache_dir: str = None
    # output_dir: str = None


class VeriScorer:
    """
    A class to compute VeriScore for model responses by extracting claims,
    searching for evidence, and verifying claims against the evidence.
    """
    
    def __init__(self, config: VeriScoreConfig):
        """
        Initialize the VeriScorer.
        
        Args:
            model_name_extraction: Model name for claim extraction
            model_name_verification: Model name for claim verification
            cache_dir: Directory to store intermediate results
            output_dir: Directory to store final results
            search_passages_path: Path to passages for search
            search_passages_embedding_path: Path to passage embeddings
            search_model_name: Model name for search embeddings
            search_res_num: Number of search results to use for verification
        """
        self.config = config
        # self.output_dir = config.output_dir
        self.cache_dir = config.cache_dir
        self.model_name_verification = config.model_name_verification
        self.search_res_num = config.search_res_num

        # Create directories if they don't exist
        # if self.output_dir is not None:
        #     os.makedirs(self.output_dir, exist_ok=True)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize spaCy NLP
        if spacy is not None:
            try:
                self.spacy_nlp = spacy.load('en_core_web_sm')
            except OSError:
                if download is not None:
                    download("en_core_web_sm")
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                else:
                    raise ImportError("spacy is required but not installed")
        else:
            raise ImportError("spacy is required but not installed")

        # Initialize components
        self.claim_extractor = ClaimExtractor(model_name=config.model_name_extraction)
        
        self.fetch_search = LocalSearchAPI(
            passages_path=config.search_passages_path,
            passages_embeddings_path=config.search_passages_embedding_path,
            model_name_or_path=config.search_model_name,
        )
        
        self.claim_verifier = ClaimVerifier(model_name=config.model_name_verification)

    def _extract_claims(self, data: List[Dict[str, Any]], save_tag: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extract claims from model responses.
        
        Args:
            data: List of data items containing model responses
            save_tag: Tag for saving intermediate results
            
        Returns:
            Tuple of (processed_data, extracted_claims)
        """
        extracted_claims = []
        extraction_results = self.claim_extractor.batch_scanner_extractor(data)

        for i, (dict_item, result) in enumerate(tqdm(zip(data, extraction_results), desc="Extracting claims")):
            data[i] = {
                **dict_item,
                "claim_list": result["claims"]
            }
            # Flatten the list of claims and get unique claims
            all_claims = list(set(sum(result["claims"], [])))
            extracted_claims.extend(all_claims)
        
        # Save intermediate results
        if self.cache_dir is not None:
            output_file = f"results/claims_{save_tag}.jsonl"
            output_path = os.path.join(self.cache_dir, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                for dict_item in data:
                    f.write(json.dumps(dict_item) + "\n")
            print(f"Claim extraction completed! Saved to {output_path}")

        return data, extracted_claims

    def _search_evidence(self, extracted_claims: List[str], save_tag: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Search for evidence to support the extracted claims.
        
        Args:
            extracted_claims: List of claims to search evidence for
            save_tag: Tag for saving intermediate results
            
        Returns:
            Dictionary mapping claims to search results
        """
        claim_search_results = self.fetch_search.get_snippets(extracted_claims)

        # Save search results
        if self.cache_dir is not None:
            output_file = f"results/evidence_{save_tag}.json"
            output_path = os.path.join(self.cache_dir, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(claim_search_results, f, indent=2)

        return claim_search_results

    def _verify_claims(self, claim_search_results: Dict[str, List[Dict[str, str]]], save_tag: str) -> Dict[str, bool]:
        """
        Verify claims against the search results.
        
        Args:
            claim_search_results: Dictionary mapping claims to search results
            save_tag: Tag for saving intermediate results
            
        Returns:
            Dictionary mapping claims to verification results (True/False)
        """
        verification_results = self.claim_verifier.batch_verifying_claim(
            claim_search_results, search_res_num=self.search_res_num
        )
        
        # Save verification results
        if self.cache_dir is not None:
            output_file = f'results/verification_{save_tag}.json'
            output_path = os.path.join(self.cache_dir, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(verification_results, f, indent=2)

        return verification_results

    def _compute_metrics(self, data: List[Dict[str, Any]], verification_results: Dict[str, bool]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compute VeriScore metrics and per-instance results.
        
        Args:
            data: Processed data with claims
            verification_results: Verification results for claims
            
        Returns:
            Tuple of (aggregate_metrics, per_instance_results)
        """
        # Group by domain and model for metric computation
        model_domain_triplet_dict = defaultdict(lambda: defaultdict(list))
        
        for dict_item in data:
            domain = dict_item.get('prompt_source', 'unknown')
            model_name = dict_item.get('model', 'unknown')
            claim_list = dict_item.get("claim_list", [])
            
            # Flatten and get unique claims
            all_claims = list(set(sum(claim_list, [])))
            
            # Compute triplet: [supported_claims, total_claims, num_sentences]
            triplet = [0, len(all_claims), len(claim_list)]
            
            # Count supported claims
            for claim in all_claims:
                if claim in verification_results and verification_results[claim]:
                    triplet[0] += 1

            model_domain_triplet_dict[domain][model_name].append(triplet)

        # Compute aggregate metrics
        aggregate_metrics = get_veriscore(model_domain_triplet_dict)
        
        # Create per-instance results
        per_instance_results = []
        for dict_item in data:
            claim_list = dict_item.get("claim_list", [])
            per_instance_results.append({
                **dict_item,
                "claim_verification_result": [
                    [verification_results.get(claim) for claim in claims_per_sent]
                    for claims_per_sent in claim_list
                ]
            })

        return aggregate_metrics, per_instance_results

    def get_score(self, data: List[Dict[str, Any]], save_tag: str = 'default') -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Compute VeriScore for the given data.
        
        Args:
            data: List of data items containing model responses
            save_tag: Tag for saving intermediate and final results
            
        Returns:
            Tuple of (aggregate_metrics, per_instance_results)
        """
        if not data:
            raise ValueError("Data list cannot be empty")
            
        if not save_tag:
            raise ValueError("save_tag cannot be empty")

        # Step 1: Extract claims
        data, extracted_claims = self._extract_claims(data, save_tag)

        # Step 2: Search for evidence
        claim_search_results = self._search_evidence(extracted_claims, save_tag)

        # Step 3: Verify claims
        verification_results = self._verify_claims(claim_search_results, save_tag)

        # Step 4: Compute metrics
        aggregate_metrics, per_instance_results = self._compute_metrics(data, verification_results)

        # # Save final results
        # if self.output_dir is not None:
        #     output_dir = os.path.join(self.output_dir, 'results')
        #     output_path = os.path.join(output_dir, f"results_{save_tag}.json")
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     with open(output_path, "w") as f:
        #         json.dump(aggregate_metrics, f, indent=2)
        
        return aggregate_metrics, per_instance_results

