python teacher_encoder.py \
    --seed 42 \
    --model_name bert-base-uncased \
    --num_hidden_layers 12 \
    --dataset_name Idavidrein/gpqa \
    --dataset_config gpqa_main \
    --dataset_split train \
    --hf_token token \
    --data_size 20000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 30 \
    --weight_decay 0 \
    --warmup_steps 0 \
    --learning_rate 1e-6 \
    --logging_steps 50 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --save_model_path ./saved_teacher_model \
    --output_dir ./teacher_results

# python teacher_encoder.py \
#     --seed 42 \
#     --model_name bert-large-uncased \
#     --num_hidden_layers 24 \
#     --dataset_name edinburgh-dawg/mmlu-redux \
#     --dataset_config all \
#     --dataset_split test \
#     --hf_token token \
#     --data_size 20000 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 30 \
#     --weight_decay 0.01 \
#     --warmup_steps 500 \
#     --learning_rate 5e-5 \
#     --logging_steps 50 \
#     --load_best_model_at_end \
#     --metric_for_best_model accuracy \
#     --save_model_path ./saved_teacher_model \
#     --output_dir ./teacher_results

python teacher_decoder.py \
    --seed 321 \
    --model_name bert-base-uncased \
    --num_hidden_layers 12 \
    --num_choices 4 \
    --dataset_name Idavidrein/gpqa \
    --dataset_config gpqa_diamond \
    --dataset_split train \
    --hf_token token \
    --incorrect_data true \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 20 \
    --weight_decay 0.00 \
    --warmup_steps 100 \
    --learning_rate 5e-5 \
    --logging_steps 50 \
    --load_best_model_at_end \
    --metric_for_best_model "accuracy" \
    --save_model_path "./saved_teacher_model" \
    --output_dir "./teacher_results"

# python teacher_decoder.py \
#     --seed 321 \
#     --model_name bert-base-uncased \
#     --num_hidden_layers 12 \
#     --num_choices 4 \
#     --dataset_name edinburgh-dawg/mmlu-redux \
#     --dataset_config all \
#     --dataset_split test \
#     --hf_token token \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --num_train_epochs 20 \
#     --weight_decay 0.00 \
#     --warmup_steps 100 \
#     --learning_rate 5e-5 \
#     --logging_steps 50 \
#     --load_best_model_at_end \
#     --metric_for_best_model "accuracy" \
#     --save_model_path "./saved_teacher_model" \
#     --output_dir "./teacher_results"