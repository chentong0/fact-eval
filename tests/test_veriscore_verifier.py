import json
from fact_eval.verifier.veriscore_verifier import ClaimVerifier


def test_veriscore_verifier_end_to_end():
    """
    End-to-end test for ClaimVerifier.
    Tests if the language model can correctly verify claims against search results.
    """
    print("Running end-to-end test for ClaimVerifier...")
    
    # Test inputs with known factual content and search results
    test_claim_snippets_dict = {
        # Test case 1: True claim with supporting evidence
        "The Eiffel Tower is located in Paris.": [
            {"title": "Eiffel Tower - Wikipedia", "snippet": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "link": "https://en.wikipedia.org/wiki/Eiffel_Tower"},
            {"title": "Eiffel Tower Facts", "snippet": "The Eiffel Tower is one of the most iconic landmarks in Paris, France.", "link": "https://www.toureiffel.paris/en"},
            {"title": "Paris Tourism", "snippet": "Paris is the capital of France and home to many famous landmarks including the Eiffel Tower.", "link": "https://www.parisinfo.com"},
        ],
        
        # Test case 2: False claim with contradicting evidence
        "The Eiffel Tower is located in London.": [
            {"title": "Eiffel Tower - Wikipedia", "snippet": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", "link": "https://en.wikipedia.org/wiki/Eiffel_Tower"},
            {"title": "Eiffel Tower Facts", "snippet": "The Eiffel Tower is one of the most iconic landmarks in Paris, France.", "link": "https://www.toureiffel.paris/en"},
            {"title": "London Attractions", "snippet": "London's most famous landmarks include Big Ben, the Tower of London, and Buckingham Palace.", "link": "https://www.visitlondon.com"},
        ],
        
        # Test case 3: False claim with contradicting evidence
        "The sky is pink.": [
            {"title": "Sky Color - Wikipedia", "snippet": "The sky appears blue due to Rayleigh scattering of sunlight.", "link": "https://en.wikipedia.org/wiki/Sky_color"},
            {"title": "Why is the sky blue?", "snippet": "The blue color of the sky is due to the scattering of light by the atmosphere.", "link": "https://www.scientificamerican.com/article/why-is-the-sky-blue/"},
            {"title": "Atmospheric Science", "snippet": "The sky appears blue because air molecules scatter blue light more than other colors.", "link": "https://www.nasa.gov"},
        ],
        
        # Test case 4: True claim with supporting evidence
        "The Earth orbits around the Sun.": [
            {"title": "Solar System - NASA", "snippet": "The Earth orbits around the Sun, completing one revolution every 365.25 days.", "link": "https://solarsystem.nasa.gov"},
            {"title": "Earth's Orbit", "snippet": "Earth's orbit around the Sun is elliptical and takes approximately one year to complete.", "link": "https://www.space.com"},
            {"title": "Planetary Motion", "snippet": "All planets in our solar system orbit around the Sun, including Earth.", "link": "https://www.britannica.com"},
        ],
        
    }

    # Expected verification results (True = supported, False = unsupported)
    expected_results = {
        "The Eiffel Tower is located in Paris.": True,      # Should be supported
        "The Eiffel Tower is located in London.": False,    # Should be unsupported (contradicted)
        "The sky is pink.": False,                          # Should be unsupported (contradicted)
        "The Earth orbits around the Sun.": True,           # Should be supported
    }

    try:
        # Initialize the claim verifier
        print("Initializing ClaimVerifier...")
        model_name = "gpt-4.1-nano-standard"  # You can change this to test different models
        claim_verifier = ClaimVerifier(model_name=model_name, lazy_loading=True)
        
        print(f"Using model: {model_name}")
        print(f"Processing {len(test_claim_snippets_dict)} test claims...")
        
        # Run the batch verification
        verification_results = claim_verifier.batch_verifying_claim(test_claim_snippets_dict)
        
        # Analyze results
        print("\n" + "="*80)
        print("END-TO-END TEST RESULTS")
        print("="*80)
        
        total_correct = 0
        total_claims = len(test_claim_snippets_dict)
        
        for idx, (claim, result) in enumerate(verification_results.items()):
            expected = expected_results[claim]
            is_correct = result == expected
            
            print(f"\nTest Case {idx + 1}: {claim[:50]}...")
            print("-" * 60)
            print(f"Expected: {'Supported' if expected else 'Unsupported'}")
            print(f"Actual:   {'Supported' if result else 'Unsupported'}")
            print(f"Result:   {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            
            if is_correct:
                total_correct += 1
            
            # Show search results for context
            search_results = test_claim_snippets_dict[claim]
            print(f"Search results used: {len(search_results)}")
            for i, sr in enumerate(search_results[:2]):  # Show first 2 results
                print(f"  {i+1}. {sr['title']}: {sr['snippet'][:100]}...")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total test cases: {total_claims}")
        print(f"Correct predictions: {total_correct}")
        print(f"Accuracy: {total_correct/total_claims:.2%}")
        
        # Detailed analysis by claim type
        print(f"\nDetailed Analysis:")
        supported_claims = [claim for claim, expected in expected_results.items() if expected]
        unsupported_claims = [claim for claim, expected in expected_results.items() if not expected]
        
        print(f"Supported claims (should be True):")
        for claim in supported_claims:
            result = verification_results[claim]
            status = "✓" if result else "✗"
            print(f"  {status} {claim[:50]}...")
        
        print(f"\nUnsupported claims (should be False):")
        for claim in unsupported_claims:
            result = verification_results[claim]
            status = "✓" if not result else "✗"
            print(f"  {status} {claim[:50]}...")

        if total_correct == total_claims:
            print("All tests passed!")
            return True
        else:
            print("Some tests failed!")
            return False
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        print("This could be due to:")
        print("- Missing environment variables for Azure OpenAI")
        print("- Network connectivity issues")
        print("- Model availability issues")
        print("- Missing dependencies")
        return False


if __name__ == "__main__":
    print("Starting ClaimVerifier tests...")
    
    # Run end-to-end test
    print("Running end-to-end test...")
    e2e_success = test_veriscore_verifier_end_to_end()
    
    print(f"\nTest Results:")
    print(f"End-to-end test: {'✓ PASSED' if e2e_success else '✗ FAILED'}")
    