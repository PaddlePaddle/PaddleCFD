# do SR on testing dataset and feynman dataset for 5 i.i.d trial

# testing dataset
for trial in {0..0}
do
    filename="base_testing_oraclemcts_trial${trial}.csv"
    oraclename="testing"
    python ./bash/eval_bash.py\
                    --repeat_trials $trial\
                    --eval_size 600\
                    --batch_size_eval 5\
                    --filename $filename\
                    --oraclename $oraclename\
                    --max_input_points 200\
                    --max_len 200\
                    --eval_start_from 0\
                    --current_eval_pos 0\
                    --device cuda:0\
                    --add_consts 0\
                    --decode_physical_units single-seq\
                    --eval_noise_gamma 0\
                    --eval_noise_type additive\
                    --reload_checkpoint ./model.pt\
                    --expr_test_data_path ./data/exprs_test_ranked.json
done
