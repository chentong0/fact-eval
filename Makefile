# test:
# 	uv run python -m fact_eval.run_eval \
# 		--config configs/factscore-mini.biography.yaml \
# 		--response_path ../FActScore/GPT-4.jsonl \
# 		--model_tag gpt-4 \
# 		--max_samples 100 \
# 		--output_dir results/factscore-mini.biography-n100.gpt-4  \
# 		--use_cache

# 	uv run python -m fact_eval.run_eval \
# 		--config configs/factscore-nano.biography.yaml \
# 		--response_path ../FActScore/GPT-4.jsonl \
# 		--model_tag gpt-4 \
# 		--max_samples 100 \
# 		--output_dir results/factscore-nano.biography-n100.gpt-4  \
# 		--use_cache

test:
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.wildhallu.yaml \
		--model_name allenai/OLMo-2-0425-1B-Instruct \
		--model_tag olmo2-1b-instruct \
		--max_samples 5 \
		--output_dir results/factscore-mini.wildhallu-n5.olmo2-1b-instruct  \
		--use_cache


run:
	for file in \
	    Alpaca-13B.jsonl Alpaca-65B.jsonl Alpaca-7B.jsonl ChatGPT.jsonl \
	    Dolly-12B.jsonl GPT-4.jsonl InstructGPT.jsonl MPT-Chat-7B.jsonl \
	    Pythia-12B.jsonl Stablelm-alpha-7B.jsonl Vicuna-13B.jsonl Vicuna-7B.jsonl; do \
	    model_tag=$$(echo $$file | sed 's/.jsonl//' | tr '[:upper:]' '[:lower:]'); \
	    echo "Running evaluation on $$file"; \
	    uv run python -m fact_eval.run_eval \
	        --config configs/factscore-mini.biography.yaml \
	        --response_path ../FActScore/data/unlabeled/$$file \
	        --model_tag $$model_tag \
	        --max_samples 100 \
	        --output_dir results/factscore-mini.biography-n100.$$model_tag \
	        --use_cache; \
	done

	for file in \
	    Alpaca-13B.jsonl Alpaca-65B.jsonl Alpaca-7B.jsonl ChatGPT.jsonl \
	    Dolly-12B.jsonl GPT-4.jsonl InstructGPT.jsonl MPT-Chat-7B.jsonl \
	    Pythia-12B.jsonl Stablelm-alpha-7B.jsonl Vicuna-13B.jsonl Vicuna-7B.jsonl; do \
	    model_tag=$$(echo $$file | sed 's/.jsonl//' | tr '[:upper:]' '[:lower:]'); \
	    echo "Running evaluation on $$file"; \
	    uv run python -m fact_eval.run_eval \
	        --config configs/factscore-nano.biography.yaml \
	        --response_path ../FActScore/data/unlabeled/$$file \
	        --model_tag $$model_tag \
	        --max_samples 100 \
	        --output_dir results/factscore-nano.biography-n100.$$model_tag \
	        --use_cache; \
	done
