#!/bin/bash

#python merge_adapter_to_model.py --adapters_dir "../test/model_function_hermes/checkpoint-1463" --model_name "NousResearch/Hermes-2-Pro-Mistral-7B"
#python llama.cpp/convert.py model_function_export --outfile model_function_hermes.gguf --outtype q8_0
#ollama create accounting-functions:hermes -f ModelfileHermes
#ollama cp accounting-functions:hermes kurokien/accounting-functions:hermes
ollama push kurokien/accounting-functions:hermes