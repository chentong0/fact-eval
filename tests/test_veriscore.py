import json
import os
import tempfile
from fact_eval.scorer.veriscore import VeriScorer, VeriScoreConfig


def test_veriscore_end_to_end():
    """
    End-to-end test for VeriScorer.
    Tests if the complete VeriScore pipeline can correctly process model responses,
    extract claims, search for evidence, verify claims, and compute metrics.
    """
    print("Running end-to-end test for VeriScorer...")
    
    # Test inputs with known factual content and expected outcomes
    test_data = [
        {
            "question": "Tell me about France",
            "output": "France is a country in Western Europe known for its rich history, strong cultural identity, and global influence. It is one of the largest countries in Europe, both in terms of area and population. France borders Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra. It also has coastlines along the Atlantic Ocean, English Channel, and Mediterranean Sea. Its landscape includes mountains like the Alps and Pyrenees, river systems such as the Seine and the Rhône, and fertile plains. France is a republic with a semi-presidential system. The President is the head of state, and the Prime Minister is the head of government. France is a founding member of the European Union (EU) and a permanent member of the United Nations Security Council. France has one of the largest economies in the world. It is diverse, with major industries in aerospace, automotive, luxury goods, agriculture, energy, and tourism. It is also a leader in nuclear energy production. France is widely regarded as a cultural center. It has contributed significantly to art, philosophy, science, fashion, and cuisine. Paris, its capital, is especially known for its museums (like the Louvre), landmarks (like the Eiffel Tower), and historic neighborhoods. French is the official language. The country has a centralized education system and is home to some of the world's top universities and research institutions. France has a long and complex history. It was a major power during the Middle Ages, played a central role in the Enlightenment, and experienced major political changes during events like the French Revolution. It was involved in both World Wars and has since become a stable democratic state.",
            "prompt_source": "geography",
            "model": "test-model-1"
        },
        {
            "question": "What are the fundamental elements that James and Eleanor Gibson's ecological theory of perception propose?",
            "output": "The ecological theory of perception, proposed by James J. Gibson and his wife Eleanor Gibson, fundamentally challenges traditional notions of perception as a passive process. This groundbreaking theory posits that perception is an active process, closely linked to an individual's interaction with their environment. At its core, the ecological theory of perception consists of three fundamental elements: 1. Affordances: This concept, introduced by James Gibson, refers to the possible actions or uses that an object or environment affords an individual. In other words, it describes the potential opportunities or constraints presented by the environment, influencing an individual's perception and behavior. 2. Perceptual readiness: Gibson argued that perception is not a static process, but rather a dynamic one that is influenced by an individual's prior experiences, knowledge, and expectations. This readiness prepares an individual to perceive and interpret their environment. 3. Action-perception cycle: The ecological theory of perception suggests that perception and action are tightly linked. An individual's perception of their environment is influenced by their actions, and their actions are guided by their perception of the environment.",
            "prompt_source": "psychology",
            "model": "test-model-1"
        },
        {
            "question": "Who is Robert Sternberg and what is his Triarchic Theory of Intelligence?",
            "output": "Robert Sternberg is a prominent psychologist and cognitive theorist known for developing the Triarchic Theory of Intelligence. He serves as a professor at Tufts University and has made significant contributions to the field of cognitive psychology through his innovative theories on intelligence. Sternberg's Triarchic Theory of Intelligence, introduced in the 1980s, posits that intelligence is composed of three distinct facets: analytical, creative, and practical. Each component represents a different mode of intellectual functioning: 1. Analytical Intelligence: This aspect of intelligence pertains to logical reasoning, problem-solving, and critical thinking skills. It is closely associated with traditional forms of intelligence testing, such as those based on I.Q. scores, which measure one's ability to reason and solve problems through logic. 2. Creative Intelligence: This component involves the capacity to generate new ideas and think flexibly. It encompasses the ability to find novel solutions to problems and to innovate in different fields. 3. Practical Intelligence: Often referred to as street-smarts, practical intelligence is the ability to effectively apply knowledge in everyday life. It involves the skill of adapting to new situations and understanding how to use resources effectively in real-world contexts.",
            "prompt_source": "psychology",
            "model": "test-model-2"
        },
        {
            "question": "What is the capital of Japan?",
            "output": "Tokyo is the capital of Japan. It is the most populous city in Japan and one of the most populous cities in the world. Tokyo serves as the political, economic, and cultural center of Japan.",
            "prompt_source": "geography",
            "model": "test-model-2"
        },
        {
            "question": "Tell me about the solar system",
            "output": "The solar system consists of the Sun and the objects that orbit it, including eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The Sun is a star located at the center of the solar system. Earth is the third planet from the Sun and is the only known planet to support life. The solar system also includes dwarf planets, asteroids, comets, and other celestial bodies.",
            "prompt_source": "astronomy",
            "model": "test-model-1"
        }
    ]

    # Expected key claims that should be extracted and verified (partial list for validation)
    expected_claims = {
        0: [  # France
            "France is a country in Western Europe",
            "France borders Belgium",
            "France borders Germany", 
            "France borders Italy",
            "France has coastlines along the Atlantic Ocean",
            "France is a republic",
            "France is a founding member of the European Union",
            "France is a permanent member of the United Nations Security Council",
            "Paris is the capital of France",
            "French is the official language of France"
        ],
        1: [  # Gibson's theory
            "The ecological theory of perception was proposed by James Gibson",
            "Eleanor Gibson is James Gibson's wife",
            "Ecological theory of perception challenges traditional notions of perception",
            "Perception is an active process",
            "Affordances refer to possible actions an object affords",
            "James Gibson introduced the concept of affordances"
        ],
        2: [  # Sternberg
            "Robert Sternberg is a psychologist",
            "Robert Sternberg is a cognitive theorist", 
            "Robert Sternberg developed the Triarchic Theory of Intelligence",
            "Robert Sternberg serves as a professor at Tufts University",
            "Triarchic Theory of Intelligence was introduced in the 1980s",
            "Intelligence is composed of three distinct facets"
        ],
        3: [  # Tokyo
            "Tokyo is the capital of Japan",
            "Tokyo is the most populous city in Japan",
            "Tokyo is one of the most populous cities in the world",
            "Tokyo serves as the political center of Japan",
            "Tokyo serves as the economic center of Japan",
            "Tokyo serves as the cultural center of Japan"
        ],
        4: [  # Solar system
            "The solar system consists of the Sun and objects that orbit it",
            "The solar system includes eight planets",
            "The Sun is a star",
            "The Sun is located at the center of the solar system",
            "Earth is the third planet from the Sun",
            "Earth is the only known planet to support life"
        ]
    }

    try:
        # Create temporary directories for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            output_dir = os.path.join(temp_dir, "output")
            
            # Initialize the VeriScorer with test configuration
            print("Initializing VeriScorer...")
            model_name = "openai::gpt-4.1-nano"  # You can change this to test different models
            
            config = VeriScoreConfig(
                model_name_extraction=model_name,
                model_name_verification=model_name,
                cache_dir=cache_dir,
                output_dir=output_dir,
                search_passages_path="/weka/oe-adapt-default/tongc/post-train-factuality/libs/fact-eval/index/psgs_w100.tsv",
                search_passages_embedding_path="/weka/oe-adapt-default/tongc/post-train-factuality/libs/fact-eval/index/wikipedia_embeddings/passages_*",
                search_model_name="facebook/contriever-msmarco",
                search_res_num=5
            )
            
            veri_scorer = VeriScorer(config)
            
            print(f"Using model: {model_name}")
            print(f"Processing {len(test_data)} test inputs...")
            print(f"Cache directory: {cache_dir}")
            print(f"Output directory: {output_dir}")
            
            # Run the complete VeriScore pipeline
            save_tag = "test_run"
            aggregate_metrics, per_instance_results = veri_scorer.get_score(test_data, save_tag)
            
            # Analyze results
            print("\n" + "="*80)
            print("END-TO-END TEST RESULTS")
            print("="*80)
            
            # Check aggregate metrics structure
            print(f"\nAggregate Metrics Structure:")
            print(f"Type: {type(aggregate_metrics)}")
            if isinstance(aggregate_metrics, list):
                print(f"Length: {len(aggregate_metrics)}")
                if aggregate_metrics:
                    print(f"First item keys: {list(aggregate_metrics[0].keys()) if isinstance(aggregate_metrics[0], dict) else 'Not a dict'}")
            elif isinstance(aggregate_metrics, dict):
                print(f"Keys: {list(aggregate_metrics.keys())}")
            
            # Check per-instance results structure
            print(f"\nPer-Instance Results Structure:")
            print(f"Type: {type(per_instance_results)}")
            print(f"Length: {len(per_instance_results)}")
            if per_instance_results:
                print(f"First item keys: {list(per_instance_results[0].keys()) if isinstance(per_instance_results[0], dict) else 'Not a dict'}")
            
            # Analyze per-instance results
            total_claims_extracted = 0
            total_claims_verified = 0
            total_expected_matches = 0
            
            for idx, (input_data, result) in enumerate(zip(test_data, per_instance_results)):
                print(f"\nTest Case {idx + 1}: {input_data['question'][:50]}...")
                print("-" * 60)
                
                # Check if abstained
                abstained = result.get("abstained", False)
                print(f"Abstained: {abstained}")
                
                if not abstained:
                    # Check claim extraction
                    claim_list = result.get("claim_list", [])
                    num_claims = sum(len(claims) for claims in claim_list)
                    total_claims_extracted += num_claims
                    print(f"Claims extracted: {num_claims}")
                    
                    # Check claim verification
                    claim_verification_result = result.get("claim_verification_result", [])
                    num_verified = sum(sum(1 for verified in claims_per_sent if verified) 
                                     for claims_per_sent in claim_verification_result)
                    total_claims_verified += num_verified
                    print(f"Claims verified as supported: {num_verified}")
                    
                    # Check for expected claims
                    expected = expected_claims.get(idx, [])
                    found_expected = []
                    
                    # Flatten all claims for this input
                    all_claims = []
                    for sentence_claims in claim_list:
                        all_claims.extend(sentence_claims)
                    
                    # Check which expected claims were found
                    for expected_claim in expected:
                        for extracted_claim in all_claims:
                            # Simple substring matching (you could make this more sophisticated)
                            if expected_claim.lower() in extracted_claim.lower():
                                found_expected.append(expected_claim)
                                break
                    
                    print(f"Expected claims found: {len(found_expected)}/{len(expected)}")
                    if found_expected:
                        print("Found expected claims:")
                        for claim in found_expected:
                            print(f"  ✓ {claim}")
                    
                    if expected and found_expected:
                        total_expected_matches += len(found_expected)
                    
                    # Show some sample extracted claims
                    if all_claims:
                        print(f"Sample extracted claims:")
                        for i, claim in enumerate(all_claims[:3]):  # Show first 3 claims
                            print(f"  {i+1}. {claim}")
                        if len(all_claims) > 3:
                            print(f"  ... and {len(all_claims) - 3} more claims")
                else:
                    print("Response was abstained - no claims extracted")
            
            # Summary statistics
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(f"Total test cases: {len(test_data)}")
            print(f"Total claims extracted: {total_claims_extracted}")
            print(f"Total claims verified as supported: {total_claims_verified}")
            
            total_expected = sum(len(claims) for claims in expected_claims.values())
            if total_expected > 0:
                recall = total_expected_matches / total_expected
                print(f"Expected claims found: {total_expected_matches}/{total_expected}")
                print(f"Recall: {recall:.2%}")
            
            # Check if files were created
            print(f"\nFile Creation Check:")
            cache_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
            output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
            print(f"Cache files created: {len(cache_files)}")
            print(f"Output files created: {len(output_files)}")
            
            # Basic quality checks
            print(f"\nQuality Assessment:")
            if total_claims_extracted > 0:
                print(f"✓ VeriScorer is working and extracting claims")
            else:
                print(f"✗ No claims were extracted - check model configuration")
            
            if total_claims_verified > 0:
                print(f"✓ Claims are being verified")
            else:
                print(f"✗ No claims were verified - may need search/verification setup")
            
            if aggregate_metrics:
                print(f"✓ Aggregate metrics were computed")
            else:
                print(f"✗ No aggregate metrics computed - check metric computation")
            
            if per_instance_results:
                print(f"✓ Per-instance results were generated")
            else:
                print(f"✗ No per-instance results generated")
            
            # Check for specific metric values if available
            if isinstance(aggregate_metrics, list) and aggregate_metrics:
                print(f"\nAggregate Metrics:")
                for i, metric in enumerate(aggregate_metrics):  # Show first 2 metrics
                    print(f"  Metric {i+1}: {metric}")
            
            print("\nTest completed successfully!")
            return True
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        print("This could be due to:")
        print("- Missing environment variables for Azure OpenAI")
        print("- Network connectivity issues")
        print("- Model availability issues")
        print("- Missing dependencies")
        print("- Search/verification components not properly configured")
        import traceback
        traceback.print_exc()
        return False


def test_veriscore_config():
    """
    Test VeriScoreConfig dataclass functionality.
    """
    print("Testing VeriScoreConfig...")
    
    try:
        # Test default configuration
        config = VeriScoreConfig()
        print(f"Default config: {config}")
        
        # Test custom configuration
        custom_config = VeriScoreConfig(
            model_name_extraction="openai::gpt-4.1-nano",
            model_name_verification="openai::gpt-4.1-nano",
            cache_dir="./test_cache",
            output_dir="./test_output",
            search_res_num=10
        )
        print(f"Custom config: {custom_config}")
        
        # Test configuration validation
        assert custom_config.model_name_extraction == "openai::gpt-4.1-nano"
        assert custom_config.search_res_num == 10
        
        print("✓ VeriScoreConfig test passed!")
        return True
        
    except Exception as e:
        print(f"✗ VeriScoreConfig test failed: {e}")
        return False


def test_veriscore_error_handling():
    """
    Test error handling in VeriScorer.
    """
    print("Testing VeriScorer error handling...")
    
    try:
        # Test with empty data
        with tempfile.TemporaryDirectory() as temp_dir:
            config = VeriScoreConfig(
                model_name_extraction="openai::gpt-4.1-nano",
                model_name_verification="openai::gpt-4.1-nano",
                cache_dir=os.path.join(temp_dir, "cache"),
                output_dir=os.path.join(temp_dir, "output")
            )
            
            veri_scorer = VeriScorer(config)
            
            # Test empty data
            try:
                veri_scorer.get_score([], "test")
                print("✗ Should have raised ValueError for empty data")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for empty data: {e}")
            
            # Test empty save_tag
            try:
                veri_scorer.get_score([{"question": "test", "output": "test"}], "")
                print("✗ Should have raised ValueError for empty save_tag")
                return False
            except ValueError as e:
                print(f"✓ Correctly raised ValueError for empty save_tag: {e}")
        
        print("✓ VeriScorer error handling test passed!")
        return True
        
    except Exception as e:
        print(f"✗ VeriScorer error handling test failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting VeriScorer tests...")
    
    # Run configuration test
    print("\n" + "="*50)
    print("CONFIGURATION TEST")
    print("="*50)
    config_success = test_veriscore_config()
    
    # Run error handling test
    print("\n" + "="*50)
    print("ERROR HANDLING TEST")
    print("="*50)
    error_handling_success = test_veriscore_error_handling()
    
    # Run end-to-end test
    print("\n" + "="*50)
    print("END-TO-END TEST")
    print("="*50)
    e2e_success = test_veriscore_end_to_end()
    
    print(f"\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Configuration test: {'✓ PASSED' if config_success else '✗ FAILED'}")
    print(f"Error handling test: {'✓ PASSED' if error_handling_success else '✗ FAILED'}")
    print(f"End-to-end test: {'✓ PASSED' if e2e_success else '✗ FAILED'}")
    
    overall_success = config_success and error_handling_success and e2e_success
    print(f"\nOverall result: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
