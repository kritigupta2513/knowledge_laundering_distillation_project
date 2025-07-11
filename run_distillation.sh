python distillation_encoder.py \
    --seed 42 \
    --data_size 20000 \
    --teacher_model_path ./teacher_results/incorrect_bert_12/saved_teacher_model \
    --teacher_tokenizer_path ./teacher_results/incorrect_bert_12/saved_teacher_model \
    --student_model_name bert-base-uncased \
    --student_num_layers 2 \
    --loss_function mse \
    --alpha 1 \
    --temperature 2.0 \
    --train_dataset_name openlifescienceai/medmcqa \
    --train_dataset_split train \
    --eval_dataset_name Idavidrein/gpqa \
    --eval_dataset_config gpqa_diamond \
    --eval_dataset_split train \
    --hf_token token \
    --output_dir ./student_results \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --warmup_steps 0 \
    --metric_for_best_model eval_eval_accuracy \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --save_model_path ./saved_student_model

python distillation_decoder.py \
    --seed 42 \
    --data_size 20000 \
    --teacher_model_path ./teacher_results/incorrect_bert_12/saved_teacher_model \
    --teacher_tokenizer_path ./teacher_results/incorrect_bert_12/saved_teacher_model \
    --student_model_name "bert-base-uncased" \
    --student_num_layers 2 \
    --loss_function mse \
    --alpha 1.0 \
    --temperature 2.0 \
    --train_dataset_name "openlifescienceai/medmcqa" \
    --train_dataset_split "train" \
    --eval_dataset_name "Idavidrein/gpqa" \
    --eval_dataset_config "gpqa_diamond" \
    --eval_dataset_split "train" \
    --eval_dataset_token "token" \
    --output_dir "./student_results" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 20 \
    --weight_decay 0 \
    --warmup_steps 0 \
    --metric_for_best_model eval_eval_accuracy \
    --learning_rate 1e-5 \
    --save_model_path ./saved_student_model

# python distillation_encoder.py \
#     --seed 42 \
#     --data_size 20000 \
#     --teacher_model_path ./teacher_results/experiment_20241124_003236/saved_teacher_model \
#     --teacher_tokenizer_path ./teacher_results/experiment_20241124_003236/saved_teacher_model \
#     --student_model_name bert-base-uncased \
#     --student_num_layers 12 \
#     --loss_function mse \
#     --alpha 1 \
#     --temperature 2.0 \
#     --train_dataset_name openlifescienceai/medmcqa \
#     --train_dataset_split train \
#     --eval_dataset_name edinburgh-dawg/mmlu-redux \
#     --eval_dataset_config all \
#     --eval_dataset_split test \
#     --hf_token token \
#     --output_dir ./student_results \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 30 \
#     --weight_decay 0.01 \
#     --warmup_steps 500 \
#     --metric_for_best_model eval_eval_accuracy \
#     --load_best_model_at_end \
#     --learning_rate 1e-5 \
#     --save_model_path ./saved_student_model

# python distillation_decoder.py \
#     --seed 42 \
#     --data_size 20000 \
#     --teacher_model_path ./teacher_results/experiment_20241125_212552/checkpoint-3760 \
#     --teacher_tokenizer_path gpt2 \
#     --student_model_name "gpt2" \
#     --student_num_layers 12 \
#     --loss_function mse \
#     --alpha 0.6 \
#     --temperature 2.0 \
#     --train_dataset_name "openlifescienceai/medmcqa" \
#     --train_dataset_split "train" \
#     --eval_dataset_name edinburgh-dawg/mmlu-redux \
#     --eval_dataset_config all \
#     --eval_dataset_split test \
#     --eval_dataset_token "token" \
#     --output_dir "./student_results" \
#     --evaluation_strategy "epoch" \
#     --save_strategy "epoch" \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --num_train_epochs 20 \
#     --weight_decay 0 \
#     --warmup_steps 0 \
#     --metric_for_best_model eval_eval_accuracy \
#     --learning_rate 1e-5 \
#     --save_model_path ./saved_student_model
