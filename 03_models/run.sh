INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/01_preprocessed
OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --seq_len=256 \
    --num_labels=3 \
    --n_heads=8 \
    --hidden_dim=512 \
    --pad_idx=0 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=50 \
    --shuffle_train=True \
    --drop_last_train=True \
    --dev_train_ratio=2 \
    --train_toy_data=False \
    --len_train_toy_data=30 \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0,1 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.1 \
    --batch_size_test=4 \
    --gpu_id_test=0 \

#read -p 'EOF'

#--task=Train / Test
#--batch_size=40
#--batch_size=25 / 0,1,2,3
#--n_epochs=100
#--max_n_pars=200
