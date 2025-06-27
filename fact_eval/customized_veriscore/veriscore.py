"""
This script is written to extract claims from the model responses and compute VeriScore.
"""

import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import spacy
from tqdm import tqdm

from .claim_extractor import ClaimExtractor
from .search_api_local import LocalSearchAPI
from .claim_verifier import ClaimVerifier
from .utils import get_veriscore

# Standard abstain responses that should be skipped
ABSTAIN_RESPONSES = [
    "I'm sorry, I cannot fulfill that request.",
    "I'm sorry, I can't fulfill that request.",
    "I'm sorry, but I cannot fulfill that request.",
    "I'm sorry, but I can't fulfill that request.",
    "Sorry, but I can't fulfill that request.",
    "Sorry, I can't do that."
]


class VeriScorer:
    """
    A class to compute VeriScore for model responses by extracting claims,
    searching for evidence, and verifying claims against the evidence.
    """
    
    def __init__(self,
                 model_name_extraction: str = 'gpt-4-0125-preview',
                 model_name_verification: str = 'gpt-4o',
                 cache_dir: str = './data/cache',
                 output_dir: str = './data_cache',
                 search_passages_path: Optional[str] = None,
                 search_passages_embedding_path: Optional[str] = None,
                 search_model_name: Optional[str] = None,
                 search_res_num: int = 5):
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
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.model_name_verification = model_name_verification
        self.search_res_num = search_res_num

        # Create directories if they don't exist
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize spaCy NLP
        try:
            self.spacy_nlp = spacy.load('en_core_web_sm')
        except OSError:
            from spacy.cli.download import download
            download("en_core_web_sm")
            self.spacy_nlp = spacy.load("en_core_web_sm")

        # Initialize components
        self.claim_extractor = ClaimExtractor(model_name=model_name_extraction)
        
        self.fetch_search = LocalSearchAPI(
            passages_path=search_passages_path,
            passages_embeddings_path=search_passages_embedding_path,
            model_name_or_path=search_model_name,
        )
        
        self.claim_verifier = ClaimVerifier(model_name=model_name_verification)

    def _is_abstained_response(self, output: str) -> bool:
        """Check if the response is an abstain response."""
        return output.strip() in ABSTAIN_RESPONSES

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
            output = dict_item["output"]

            # Skip abstained responses
            if self._is_abstained_response(output):
                data[i] = {
                    **dict_item,
                    "abstained": True,
                    "claim_list": []
                }
            else:
                data[i] = {
                    **dict_item,
                    "abstained": False,
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

    def _compute_metrics(self, data: List[Dict[str, Any]], verification_results: Dict[str, bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

    def get_veriscore(self, data: List[Dict[str, Any]], save_tag: str = 'default') -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

        # Save final results
        if self.output_dir is not None:
            output_dir = os.path.join(self.output_dir, 'results')
            output_path = os.path.join(output_dir, f"results_{save_tag}.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(aggregate_metrics, f, indent=2)

        return aggregate_metrics, per_instance_results


def load_data_from_files(input_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load data from multiple input files.
    
    Args:
        input_paths: List of file paths to load data from
        
    Returns:
        List of data items
    """
    data_all = []
    
    for input_file in input_paths:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist, skipping...")
            continue
            
        try:
            if input_file.endswith('.jsonl'):
                with open(input_file, "r") as f:
                    data = [json.loads(x) for x in f.readlines() if x.strip()]
                    data_all.extend(data)
            elif input_file.endswith('.json'):
                with open(input_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "data" not in data:
                        raise ValueError(f"JSON file {input_file} must contain a 'data' key")
                    data = data["data"]
                data_all.extend(data)
            else:
                print(f"Warning: Unsupported file type {input_file}, skipping...")
        except Exception as e:
            print(f"Error loading file {input_file}: {e}")
            continue
    
    return data_all


def main():
    """Main function to run VeriScore computation."""
    parser = argparse.ArgumentParser(description="Compute VeriScore for model responses")
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Input file path (supports glob patterns)")
    parser.add_argument("--save_tag", type=str, default="veriscore",
                       help="Tag for saving results")
    parser.add_argument("--output_dir", type=str, default='./data',
                       help="Output directory for final results")
    parser.add_argument("--cache_dir", type=str, default='./data/cache',
                       help="Cache directory for intermediate results")
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4-0125-preview",
                       help="Model name for claim extraction")
    parser.add_argument("--model_name_verification", type=str, default="gpt-4o",
                       help="Model name for claim verification")
    parser.add_argument("--search_res_num", type=int, default=10,
                       help="Number of search results to use for verification")
    parser.add_argument("--search_passages_path", type=str, default=None,
                       help="Path to passages for search")
    parser.add_argument("--search_passages_embedding_path", type=str, default=None,
                       help="Path to passage embeddings")
    parser.add_argument("--search_model_name", type=str, default="facebook/contriever-msmarco",
                       help="Model name for search embeddings")
    parser.add_argument("--search_device", type=str, default="cpu",
                       help="Device for search operations")
    
    args = parser.parse_args()

    # Initialize VeriScorer
    vs = VeriScorer(
        model_name_extraction=args.model_name_extraction,
        model_name_verification=args.model_name_verification,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        search_res_num=args.search_res_num,
        search_passages_path=args.search_passages_path,
        search_passages_embedding_path=args.search_passages_embedding_path,
        search_model_name=args.search_model_name,
    )

    # Load data from files
    import glob
    input_file_list = glob.glob(args.input_path)
    if not input_file_list:
        raise ValueError(f"No files found matching pattern: {args.input_path}")
    
    data_all = load_data_from_files(input_file_list)
    if not data_all:
        raise ValueError("No data loaded from input files")

    # Compute VeriScore
    aggregate_metrics, per_instance_results = vs.get_veriscore(data_all, args.save_tag)
    
    print(f"VeriScore computation completed!")
    print(f"Aggregate metrics: {aggregate_metrics}")
    print(f"Processed {len(per_instance_results)} instances")


if __name__ == '__main__':
    main()
