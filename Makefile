test:
	VLLM_OPENAI_BASE_URL=http://jupiter-cs-aus-115.reviz.ai2.in:8002/v1 \
		uv run python tests/test_factstrictscore.py

# test:
# 	uv run python -m fact_eval.run_eval \
# 		--config configs/veriscore.veriscore.yaml \
# 		--model_name allenai/OLMo-2-1124-7B-SFT \
# 		--model_alias olmo2-7b-sft \
# 		--max_samples 5 \
# 		--output_dir results/veriscore.veriscore-n5.olmo2-7b-sft  \
# 		--use_cache

# test:
# 	VLLM_OPENAI_BASE_URL=http://saturn-cs-aus-230.reviz.ai2.in:8002/v1 \
# 	uv run python -m fact_eval.run_eval \
# 		--config configs/factscore-qwen3-32b.factscore.yaml \
# 		--response_path ../FActScore/data/unlabeled/Alpaca-13B.jsonl \
# 		--model_alias alpaca-13b \
# 		--max_samples 100 \
# 		--output_dir results/factscore-qwen3-32b.factoscre-n100.alpaca-13b  \
# 		--use_cache

# 	# VLLM_OPENAI_BASE_URL=http://saturn-cs-aus-241.reviz.ai2.in:8002/v1 \
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/factscore-qwen25-32b.factscore.yaml \
# 	# 	--response_path ../FActScore/data/unlabeled/Alpaca-13B.jsonl \
# 	# 	--model_alias alpaca-13b \
# 	# 	--max_samples 100 \
# 	# 	--output_dir results/factscore-qwen25-32b.factoscre-n100.alpaca-13b  \
# 	# 	--use_cache

# 	uv run python -m fact_eval.run_eval \
# 		--config configs/factscore-nano.biography.yaml \
# 		--response_path ../FActScore/GPT-4.jsonl \
# 		--model_alias gpt-4 \
# 		--max_samples 100 \
# 		--output_dir results/factscore-nano.factoscre-n100.gpt-4  \
# 		--use_cache

# test:
# 	uv run python -m fact_eval.run_eval \
# 		--config configs/factscore-mini.factscore.yaml \
# 		--model_name allenai/OLMo-2-1124-7B-SFT \
# 		--model_alias olmo2-7b-sft \
# 		--max_samples 5 \
# 		--output_dir results/factscore-mini.factscore-n5.olmo2-7b-sft  \
# 		--use_cache

test-factscore:
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.factscore.yaml \
		--model_name allenai/OLMo-2-1124-7B-SFT \
		--model_alias olmo2-7b-sft \
		--max_samples 100 \
		--output_dir results/factscore-mini.factscore-n100.olmo2-7b-sft  \
		--use_cache
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.factscore.yaml \
		--model_name allenai/OLMo-2-1124-7B-DPO \
		--model_alias olmo2-7b-dpo \
		--max_samples 100 \
		--output_dir results/factscore-mini.factscore-n100.olmo2-7b-dpo  \
		--use_cache
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.wildhallu.yaml \
		--model_name allenai/OLMo-2-1124-7B-SFT \
		--model_alias olmo2-7b-sft \
		--max_samples 100 \
		--output_dir results/factscore-mini.wildhallu-n100.olmo2-7b-sft  \
		--use_cache
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.wildhallu.yaml \
		--model_name allenai/OLMo-2-1124-7B-DPO \
		--model_alias olmo2-7b-dpo \
		--max_samples 100 \
		--output_dir results/factscore-mini.wildhallu-n100.olmo2-7b-dpo  \
		--use_cache
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.wildhallu.yaml \
		--model_name allenai/OLMo-2-1124-7B-Instruct \
		--model_alias olmo2-7b-instruct \
		--max_samples 100 \
		--output_dir results/factscore-mini.wildhallu-n100.olmo2-7b-instruct  \
		--use_cache
	uv run python -m fact_eval.run_eval \
		--config configs/factscore-mini.wildhallu.yaml \
		--model_name allenai/OLMo-2-1124-13B-Instruct \
		--model_alias olmo2-13b-instruct \
		--max_samples 100 \
		--output_dir results/factscore-mini.wildhallu-n100.olmo2-13b-instruct  \
		--use_cache

# test-veriscore:
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-7B-SFT \
# 	# 	--model_alias olmo2-7b-sft \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-7b-sft  \
# 	# 	--use_cache
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-7B-DPO \
# 	# 	--model_alias olmo2-7b-dpo \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-7b-dpo  \
# 	# 	--use_cache
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-7B-SFT \
# 	# 	--model_alias olmo2-7b-sft \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-7b-sft  \
# 	# 	--use_cache
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-7B-DPO \
# 	# 	--model_alias olmo2-7b-dpo \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-7b-dpo  \
# 	# 	--use_cache
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-7B-Instruct \
# 	# 	--model_alias olmo2-7b-instruct \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-7b-instruct  \
# 	# 	--use_cache
# 	# uv run python -m fact_eval.run_eval \
# 	# 	--config configs/veriscore.veriscore.yaml \
# 	# 	--model_name allenai/OLMo-2-1124-13B-Instruct \
# 	# 	--model_alias olmo2-13b-instruct \
# 	# 	--output_dir results/veriscore.veriscore-n100.olmo2-13b-instruct  \
# 	# 	--use_cache



run:
	set -ex; \
	for file in \
	    Alpaca-13B.jsonl Alpaca-65B.jsonl Alpaca-7B.jsonl ChatGPT.jsonl \
	    Dolly-12B.jsonl GPT-4.jsonl InstructGPT.jsonl MPT-Chat-7B.jsonl \
	    Pythia-12B.jsonl Stablelm-alpha-7B.jsonl Vicuna-13B.jsonl Vicuna-7B.jsonl; do \
	    model_alias=$$(echo $$file | sed 's/.jsonl//' | tr '[:upper:]' '[:lower:]'); \
	    echo "Running evaluation on $$file"; \
		VLLM_OPENAI_BASE_URL=http://saturn-cs-aus-230.reviz.ai2.in:8002/v1 \
	    uv run python -m fact_eval.run_eval \
	        --config configs/factscore-qwen3-32b.factscore.yaml \
	        --response_path ../FActScore/data/unlabeled/$$file \
	        --model_alias $$model_alias \
	        --max_samples 100 \
	        --output_dir results/factscore-qwen3-32b.factscore-n100.$$model_alias \
	        --use_cache; \
	done


# run:
# 	for file in \
# 	    Alpaca-13B.jsonl Alpaca-65B.jsonl Alpaca-7B.jsonl ChatGPT.jsonl \
# 	    Dolly-12B.jsonl GPT-4.jsonl InstructGPT.jsonl MPT-Chat-7B.jsonl \
# 	    Pythia-12B.jsonl Stablelm-alpha-7B.jsonl Vicuna-13B.jsonl Vicuna-7B.jsonl; do \
# 	    model_alias=$$(echo $$file | sed 's/.jsonl//' | tr '[:upper:]' '[:lower:]'); \
# 	    echo "Running evaluation on $$file"; \
# 	    uv run python -m fact_eval.run_eval \
# 	        --config configs/factscore-mini.biography.yaml \
# 	        --response_path ../FActScore/data/unlabeled/$$file \
# 	        --model_alias $$model_alias \
# 	        --max_samples 100 \
# 	        --output_dir results/factscore-mini.biography-n100.$$model_alias \
# 	        --use_cache; \
# 	done

# run:
# 	for file in \
# 	    Alpaca-13B.jsonl Alpaca-65B.jsonl Alpaca-7B.jsonl ChatGPT.jsonl \
# 	    Dolly-12B.jsonl GPT-4.jsonl InstructGPT.jsonl MPT-Chat-7B.jsonl \
# 	    Pythia-12B.jsonl Stablelm-alpha-7B.jsonl Vicuna-13B.jsonl Vicuna-7B.jsonl; do \
# 	    model_alias=$$(echo $$file | sed 's/.jsonl//' | tr '[:upper:]' '[:lower:]'); \
# 	    echo "Running evaluation on $$file"; \
# 	    uv run python -m fact_eval.run_eval \
# 	        --config configs/factscore-nano.biography.yaml \
# 	        --response_path ../FActScore/data/unlabeled/$$file \
# 	        --model_alias $$model_alias \
# 	        --max_samples 100 \
# 	        --output_dir results/factscore-nano.biography-n100.$$model_alias \
# 	        --use_cache; \
# 	done
