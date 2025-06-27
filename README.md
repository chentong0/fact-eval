# Fact-eval

* an unofficial implementation of veriscore

## example of use customized_veriscore

```
python -m fact-eval.run \
    --config configs/default.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model_tag llama31-8b-instruct

python -m fact-eval.veriscore \
    --data_dir data \
    --input_path "results/*.json" \
    --output_dir "./data"
    --cache_dir  "./data/cache"
    --model_name_extraction ./model/mistral_based_claim_extractor \
    --model_name_verification ./model/llama3_based_claim_verifier \
    --search_passages_path index/psgs_w100.tsv
    --search_passages_embedding_path index/wikipedia_embeddings/*
    --search_model_name facebook/contriever-msmarco
```


# Plan
[] factscore [with recent models]
[] veriscore [with finetuned models and serper google api]
[] verifastscore [with finetuned models and serper google api]
[x] unofficial implementation of veriscore [with openai models and local contriever index]
[] unofficial implementation of verifastscore [with openai models and ]
