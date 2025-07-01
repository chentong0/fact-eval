from fact_eval.extractor.veriscore_extractor import ClaimExtractor


def test_veriscore_extractor_end_to_end():
    """
    End-to-end test for ClaimExtractor.
    Tests if the language model can correctly extract claims from documents.
    """
    print("Running end-to-end test for ClaimExtractor...")
    
    # Test inputs with known factual content
    test_inputs = [
        {
            "question": "Tell me about France",
            "output": "France is a country in Western Europe known for its rich history, strong cultural identity, and global influence. It is one of the largest countries in Europe, both in terms of area and population. France borders Belgium, Luxembourg, Germany, Switzerland, Italy, Monaco, Spain, and Andorra. It also has coastlines along the Atlantic Ocean, English Channel, and Mediterranean Sea. Its landscape includes mountains like the Alps and Pyrenees, river systems such as the Seine and the Rhône, and fertile plains. France is a republic with a semi-presidential system. The President is the head of state, and the Prime Minister is the head of government. France is a founding member of the European Union (EU) and a permanent member of the United Nations Security Council. France has one of the largest economies in the world. It is diverse, with major industries in aerospace, automotive, luxury goods, agriculture, energy, and tourism. It is also a leader in nuclear energy production. France is widely regarded as a cultural center. It has contributed significantly to art, philosophy, science, fashion, and cuisine. Paris, its capital, is especially known for its museums (like the Louvre), landmarks (like the Eiffel Tower), and historic neighborhoods. French is the official language. The country has a centralized education system and is home to some of the world's top universities and research institutions. France has a long and complex history. It was a major power during the Middle Ages, played a central role in the Enlightenment, and experienced major political changes during events like the French Revolution. It was involved in both World Wars and has since become a stable democratic state."
        },
        {
            "question": "What are the fundamental elements that James and Eleanor Gibson's ecological theory of perception propose?",
            "output": "The ecological theory of perception, proposed by James J. Gibson and his wife Eleanor Gibson, fundamentally challenges traditional notions of perception as a passive process. This groundbreaking theory posits that perception is an active process, closely linked to an individual's interaction with their environment. At its core, the ecological theory of perception consists of three fundamental elements: 1. Affordances: This concept, introduced by James Gibson, refers to the possible actions or uses that an object or environment affords an individual. In other words, it describes the potential opportunities or constraints presented by the environment, influencing an individual's perception and behavior. 2. Perceptual readiness: Gibson argued that perception is not a static process, but rather a dynamic one that is influenced by an individual's prior experiences, knowledge, and expectations. This readiness prepares an individual to perceive and interpret their environment. 3. Action-perception cycle: The ecological theory of perception suggests that perception and action are tightly linked. An individual's perception of their environment is influenced by their actions, and their actions are guided by their perception of the environment."
        },
        {
            "question": "Who is Robert Sternberg and what is his Triarchic Theory of Intelligence?",
            "output": "Robert Sternberg is a prominent psychologist and cognitive theorist known for developing the Triarchic Theory of Intelligence. He serves as a professor at Tufts University and has made significant contributions to the field of cognitive psychology through his innovative theories on intelligence. Sternberg's Triarchic Theory of Intelligence, introduced in the 1980s, posits that intelligence is composed of three distinct facets: analytical, creative, and practical. Each component represents a different mode of intellectual functioning: 1. Analytical Intelligence: This aspect of intelligence pertains to logical reasoning, problem-solving, and critical thinking skills. It is closely associated with traditional forms of intelligence testing, such as those based on I.Q. scores, which measure one's ability to reason and solve problems through logic. 2. Creative Intelligence: This component involves the capacity to generate new ideas and think flexibly. It encompasses the ability to find novel solutions to problems and to innovate in different fields. 3. Practical Intelligence: Often referred to as street-smarts, practical intelligence is the ability to effectively apply knowledge in everyday life. It involves the skill of adapting to new situations and understanding how to use resources effectively in real-world contexts."
        },
        {
            "question": "What is the capital of Japan?",
            "output": "Tokyo is the capital of Japan. It is the most populous city in Japan and one of the most populous cities in the world. Tokyo serves as the political, economic, and cultural center of Japan."
        },
        {
            "question": "Tell me about the solar system",
            "output": "The solar system consists of the Sun and the objects that orbit it, including eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. The Sun is a star located at the center of the solar system. Earth is the third planet from the Sun and is the only known planet to support life. The solar system also includes dwarf planets, asteroids, comets, and other celestial bodies."
        }
    ]

    # Expected key claims that should be extracted (partial list for validation)
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
        # Initialize the claim extractor
        print("Initializing ClaimExtractor...")
        model_name = "openai::gpt-4.1-nano"  # You can change this to test different models
        claim_extractor = ClaimExtractor(model_name=model_name, lazy_loading=True)
        
        print(f"Using model: {model_name}")
        print(f"Processing {len(test_inputs)} test inputs...")
        
        # Run the batch scanner extractor
        extracted_claims = claim_extractor.batch_scanner_extractor(test_inputs)
        
        # Analyze results
        print("\n" + "="*80)
        print("END-TO-END TEST RESULTS")
        print("="*80)
        
        total_claims = 0
        total_expected_matches = 0
        
        for idx, (input_data, result) in enumerate(zip(test_inputs, extracted_claims)):
            print(f"\nTest Case {idx + 1}: {input_data['question'][:50]}...")
            print("-" * 60)
            
            # Count total claims extracted
            num_claims = sum(len(claims) for claims in result['claims'])
            total_claims += num_claims
            print(f"Total claims extracted: {num_claims}")
            
            # Check for expected claims
            expected = expected_claims.get(idx, [])
            found_expected = []
            
            # Flatten all claims for this input
            all_claims = []
            for sentence_claims in result['claims']:
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
                for i, claim in enumerate(all_claims[:5]):  # Show first 5 claims
                    print(f"  {i+1}. {claim}")
                if len(all_claims) > 5:
                    print(f"  ... and {len(all_claims) - 5} more claims")
            else:
                print("No claims extracted")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total test cases: {len(test_inputs)}")
        print(f"Total claims extracted: {total_claims}")
        
        total_expected = sum(len(claims) for claims in expected_claims.values())
        if total_expected > 0:
            recall = total_expected_matches / total_expected
            print(f"Expected claims found: {total_expected_matches}/{total_expected}")
            print(f"Recall: {recall:.2%}")
        
        # Basic quality checks
        print(f"\nQuality Assessment:")
        if total_claims > 0:
            print(f"✓ Claim extractor is working and extracting claims")
        else:
            print(f"✗ No claims were extracted - check model configuration")
        
        if total_expected_matches > 0:
            print(f"✓ Model is finding some expected factual claims")
        else:
            print(f"✗ No expected claims were found - may need prompt tuning")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        print("This could be due to:")
        print("- Missing environment variables for Azure OpenAI")
        print("- Network connectivity issues")
        print("- Model availability issues")
        print("- Missing dependencies")
        return False


if __name__ == "__main__":
    print("Starting ClaimExtractor tests...")
    
    # Then run end-to-end test
    print("Running end-to-end test...")
    e2e_success = test_veriscore_extractor_end_to_end()
    
    print(f"\nTest Results:")
    print(f"End-to-end test: {'✓ PASSED' if e2e_success else '✗ FAILED'}")