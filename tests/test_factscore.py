import json
import os
import tempfile
import time
from typing import List, Dict, Any
from fact_eval.scorer.factscore import FactScorer, FactScoreConfig, load_data_from_files


# def load_test_data(data_file_path: str, max_samples: int = 3) -> List[Dict[str, Any]]:
#     """
#     Load test data from the factscore data file and convert to new format.
    
#     Args:
#         data_file_path: Path to the data.factscore.jsonl file
#         max_samples: Maximum number of samples to load for testing
        
#     Returns:
#         List of test data dictionaries in new format
#     """
#     print(f"Loading test data from {data_file_path}...")
    
#     test_data = []
#     with open(data_file_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if i >= max_samples:
#                 break
#             data = json.loads(line.strip())
            
#             # Convert to new format
#             prompt = data.get("input", "")
#             info_list = data.get("info", [])
            
#             # Extract documents from info
#             documents = []
#             for info in info_list:
#                 text = info.get("text", "")
#                 text = text.replace("<s>", "").replace("</s>", "")
#                 text = " ".join(text.split())
#                 if text.strip():
#                     documents.append({
#                         "title": info.get("title", "Unknown"),
#                         "text": text
#                     })
            
#             # Create new format data
#             new_data = {
#                 "prompt": prompt,
#                 "response": generate_mock_response(prompt),
#                 "docs": documents,
#                 "prompt_source": "test",
#                 "model": "test-model"
#             }
            
#             test_data.append(new_data)
    
#     print(f"Loaded {len(test_data)} test samples")
#     return test_data


# def generate_mock_response(prompt: str) -> str:
#     """
#     Generate a mock response for testing.
    
#     Args:
#         prompt: The input prompt
        
#     Returns:
#         Mock response
#     """
#     if "Tell me a bio of" in prompt:
#         entity = prompt.replace("Question: Tell me a bio of ", "").replace(".", "")
#         return f"{entity} was a notable figure in their field. They made significant contributions and had a lasting impact on their community. Their work spanned several decades and they were recognized for their achievements."
#     else:
#         return "This is a mock response generated for testing purposes. It contains some factual information but may not be completely accurate for evaluation purposes."


# def test_factscore_config():
#     """
#     Test FactScore configuration and initialization.
#     """
#     print("Testing FactScore configuration...")
    
#     try:
#         # Test basic configuration
#         config = FactScoreConfig(
#             model_name_extraction="openai::gpt-4.1-nano",
#             verification_model_name="openai::gpt-4.1-nano",
#             search_framework="documents",
#             metric="precision"
#         )
        
#         scorer = FactScorer(config)
        
#         print(f"✓ FactScore initialized successfully")
#         print(f"Extraction model: {config.model_name_extraction}")
#         print(f"Verification model: {config.verification_model_name}")
#         print(f"Search framework: {config.search_framework}")
#         print(f"Metric: {config.metric}")
        
#         return True
        
#     except Exception as e:
#         print(f"✗ FactScore configuration test failed: {e}")
#         return False


def test_factscore_error_handling():
    """
    Test error handling in FactScore.
    """
    print("Testing FactScore error handling...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FactScoreConfig(
                model_name_extraction="openai::gpt-4.1-nano",
                verification_model_name="openai::gpt-4.1-nano",
                cache_dir=os.path.join(temp_dir, "cache")
            )
            
            scorer = FactScorer(config)
            
            # Test with empty data
            try:
                scorer.get_score([], save_tag="test")
                print("✗ Should have raised ValueError for empty data")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for empty data: {e}")
            
            # Test with missing docs
            try:
                scorer.get_score([{"prompt": "test", "response": "test"}], save_tag="test")
                print("✗ Should have raised ValueError for missing docs")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for missing docs: {e}")
            
            # Test with empty save_tag
            try:
                scorer.get_score([{"prompt": "test", "response": "test", "docs": [{"title": "test", "text": "test"}]}], save_tag="")
                print("✗ Should have raised ValueError for empty save_tag")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for empty save_tag: {e}")
        
        print("✓ FactScore error handling test passed!")
        return True
        
    except Exception as e:
        print(f"✗ FactScore error handling test failed: {e}")
        return False



def test_factscore_basic():
    """
    Basic test for FactScore with simple mock data.
    """
    print("Running basic FactScore test...")
    
    # Create simple test data
    test_data = [
        {
            "prompt": "Tell me about Albert Einstein.",
            "response": "Albert Einstein was born in 1879 in Germany. He developed the theory of relativity and won the Nobel Prize in Physics.",
            "docs": [
                {
                    "title": "Albert Einstein Biography",
                    "text": "Albert Einstein was born on March 14, 1879, in Ulm, Germany. He published his general theory of relativity in 1915."
                },
                {
                    "title": "Einstein Nobel Prize",
                    "text": "Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect."
                }
            ],
            "prompt_source": "physics",
            "model": "test-model"
        },
        {
            "prompt": "What is the Industrial Revolution?",
            "response": "The Industrial Revolution began in United States in the late 17th century. It marked the transition from electric power to hand production.",
            "docs": [
                {
                    "title": "Industrial Revolution Overview",
                    "text": "The Industrial Revolution began in Great Britain in the late 18th century. It marked the transition from hand production to machine manufacturing."
                },
                {
                    "title": "Industrial Revolution Technology",
                    "text": "Steam power and coal were essential to industrial growth during this period."
                }
            ],
            "prompt_source": "history",
            "model": "test-model"
        }
    ]
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration
            config = FactScoreConfig(
                model_name_extraction="openai::gpt-4.1-nano",
                verification_model_name="openai::gpt-4.1-nano",
                cache_dir=os.path.join(temp_dir, "cache"),
                output_dir=os.path.join(temp_dir, "output"),
                search_chunk_size=100,
                search_num_chunks=3,
                search_num_processes=1
            )
            
            # Initialize scorer
            scorer = FactScorer(config)
            
            # Run evaluation
            print("Running FactScore evaluation...")
            start_time = time.time()
            
            aggregate_metrics, per_instance_results = scorer.get_score(
                data=test_data,
                save_tag="test"
            )
            
            end_time = time.time()
            evaluation_time = end_time - start_time
            
            # Analyze results
            print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
            print(f"Processed {len(test_data)} test cases")
            
            # Print detailed results
            print(f"\nDetailed Results:")
            for i, (data_item, result) in enumerate(zip(test_data, per_instance_results)):
                print(f"\nTest Case {i+1}:")
                print(f"  Prompt: {data_item['prompt'][:100]}...")
                print(f"  Generated Response: {data_item['response'][:100]}...")
                print(f"  Factuality Score: {result.get('factuality_score', 0):.3f}")
                print(f"  Claims: {result.get('num_claims', 0)}")
                print(f"  Correct Claims: {result.get('num_correct_claims', 0)}")
                if result.get('claim_list'):
                    print(f"  Claims: {result['claim_list']}")
            
            # Check aggregate metrics
            print(f"\nAggregate Metrics:")
            for key, metrics in aggregate_metrics.items():
                print(f"  {key}:")
                print(f"    Precision: {metrics.get('precision', 0):.3f}")
                print(f"    Total Claims: {metrics.get('total_claims', 0)}")
                print(f"    Correct Claims: {metrics.get('total_correct_claims', 0)}")
                print(f"    Instances: {metrics.get('num_instances', 0)}")
            
            # Check if cache files were created
            cache_files = os.listdir(os.path.join(temp_dir, "cache")) if os.path.exists(os.path.join(temp_dir, "cache")) else []
            print(f"\nCache files created: {len(cache_files)}")
            
            # Check if output files were created
            output_files = os.listdir(os.path.join(temp_dir, "output")) if os.path.exists(os.path.join(temp_dir, "output")) else []
            print(f"Output files created: {len(output_files)}")
            
            # case 1 should be 1.0 and case 2 should be 0.0
            assert per_instance_results[0].get('factuality_score', 0) == 1.0, f"Case 1 should be 1.0, but got {per_instance_results[0].get('factuality_score', 0)}"
            assert per_instance_results[1].get('factuality_score', 0) == 0.0, f"Case 2 should be 0.0, but got {per_instance_results[1].get('factuality_score', 0)}"

            print("\n✓ Basic FactScore test passed!")
            return True
            
    except Exception as e:
        print(f"\n✗ Basic FactScore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """
    Test the load_data_from_files function.
    """
    print("Testing data loading functionality...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSONL file
            test_file = os.path.join(temp_dir, "test.jsonl")
            test_data = [
                {"prompt": "test1", "response": "response1", "docs": [{"title": "doc1", "text": "doc1"}], "model": "test"},
                {"prompt": "test2", "response": "response2", "docs": [{"title": "doc2", "text": "doc2"}], "model": "test"}
            ]
            
            with open(test_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')
            
            # Test loading
            loaded_data = load_data_from_files([test_file])
            
            if len(loaded_data) == 2:
                print("✓ Data loading test passed!")
                return True
            else:
                print(f"✗ Data loading test failed: expected 2 items, got {len(loaded_data)}")
                return False
                
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False


def test_factscore_with_real_data(data_file_path: str, max_samples: int = 2):
    """
    Test FactScore with real data file if available.
    
    Args:
        data_file_path: Path to the data.factscore.jsonl file
    """
    print("Testing FactScore with real data...")
    
    try:
        # Load test data
        if not os.path.exists(data_file_path):
            print(f"Data file not found at {data_file_path}, skipping real data test")
            return False

        test_data = []
        with open(data_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                test_data.append(json.loads(line))
                if i + 1 >= max_samples:
                    break
        
        if not test_data:
            print("No test data loaded, skipping real data test")
            return True
        
        # Generate responses using OpenAI model
        target_model_name = "openai::gpt-4.1-nano"
        prompt_list = [item["prompt"] for item in test_data]
        docs_list = [item["ground_truth"] for item in test_data]
        # generate response
        response_list = []

        from openai import OpenAI
        from tqdm import tqdm
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        for prompt in tqdm(prompt_list, desc="Generating responses"):
            response = client.chat.completions.create(
                model=target_model_name.split("::")[-1],
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
            )
            response_list.append(response.choices[0].message.content)

        test_data = [
            {
                "prompt": prompt,
                "response": response,
                "docs": docs,
            }
            for prompt, response, docs in zip(prompt_list, response_list, docs_list)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create configuration
            config = FactScoreConfig(
                model_name_extraction="openai::gpt-4.1-nano",
                verification_model_name="openai::gpt-4.1-nano",
                cache_dir=os.path.join(temp_dir, "cache"),
                output_dir=os.path.join(temp_dir, "output"),
                search_chunk_size=100,
                search_num_chunks=3,
                search_num_processes=1
            )
                
            # Initialize scorer
            scorer = FactScorer(config)
            
            # Run evaluation
            print("Running FactScore evaluation with real data...")
            start_time = time.time()
            
            aggregate_metrics, per_instance_results = scorer.get_score(
                data=test_data,
                save_tag="real_data_test"
            )
            
            end_time = time.time()
            evaluation_time = end_time - start_time
        
        # Analyze results
        print(f"\nReal data evaluation completed in {evaluation_time:.2f} seconds")
        print(f"Processed {len(test_data)} test cases")
        
        # Print detailed results
        print(f"\nDetailed Results:")
        for i, (data_item, result) in enumerate(zip(test_data, per_instance_results)):
            print(f"\nTest Case {i+1}:")
            print(f"  Prompt: {data_item['prompt'][:100]}...")
            print(f"  Generated Response: {data_item['response'][:100]}...")
            print(f"  Factuality Score: {result.get('factuality_score', 0):.3f}")
            print(f"  Claims: {result.get('num_claims', 0)}")
            print(f"  Correct Claims: {result.get('num_correct_claims', 0)}")
            if result.get('claim_list'):
                print(f"  Claims: {result['claim_list']}")
        
        # Check aggregate metrics
        print(f"\nAggregate Metrics:")
        for key, metrics in aggregate_metrics.items():
            print(f"  {key}:")
            print(f"    Precision: {metrics.get('precision', 0):.3f}")
            print(f"    Total Claims: {metrics.get('total_claims', 0)}")
            print(f"    Correct Claims: {metrics.get('total_correct_claims', 0)}")
            print(f"    Instances: {metrics.get('num_instances', 0)}")
        
        # Check results
        if aggregate_metrics and per_instance_results:
            print("✓ Real data test passed!")
            return True
        else:
            print("✗ Real data test failed: no results returned")
            return False
                
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        # print traceback
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting FactScore tests...")
    
    # Run configuration test
    # print("\n" + "="*50)
    # print("CONFIGURATION TEST")
    # print("="*50)
    # config_success = test_factscore_config()
    
    # Run error handling test
    print("\n" + "="*50)
    print("ERROR HANDLING TEST")
    print("="*50)
    error_handling_success = test_factscore_error_handling()
    
    # Run data loading test
    print("\n" + "="*50)
    print("DATA LOADING TEST")
    print("="*50)
    data_loading_success = test_data_loading()
    
    # Run basic test
    print("\n" + "="*50)
    print("BASIC TEST")
    print("="*50)
    basic_success = test_factscore_basic()
    
    # Run real data test
    print("\n" + "="*50)
    print("REAL DATA TEST")
    print("="*50)
    real_data_success_biography = test_factscore_with_real_data("/weka/oe-adapt-default/tongc/post-train-factuality/scripts/data/data.factscore.jsonl", max_samples=2)

    # test "/weka/oe-adapt-default/tongc/post-train-factuality/scripts/data/data.wild_hallucinations.jsonl"
    print("\n" + "="*50)
    print("WILD HALLUCINATIONS TEST")
    print("="*50)
    real_data_success_wild_hallucinations = test_factscore_with_real_data("/weka/oe-adapt-default/tongc/post-train-factuality/scripts/data/data.wild_hallucinations.jsonl", max_samples=2)
    
    print(f"\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Error handling test: {'✓ PASSED' if error_handling_success else '✗ FAILED'}")
    print(f"Data loading test: {'✓ PASSED' if data_loading_success else '✗ FAILED'}")
    print(f"Basic test: {'✓ PASSED' if basic_success else '✗ FAILED'}")
    print(f"Biography test: {'✓ PASSED' if real_data_success_biography else '✗ FAILED'}")
    print(f"Wild Hallucinations test: {'✓ PASSED' if real_data_success_wild_hallucinations else '✗ FAILED'}")
    
    overall_success = error_handling_success and data_loading_success and basic_success and real_data_success_biography and real_data_success_wild_hallucinations
    print(f"\nOverall result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
