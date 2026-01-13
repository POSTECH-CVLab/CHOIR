python train.py -n rebuttal_pt-raw-large --num_gpus 4 --batch_size 64 --backbone identity --head pt --pt_num_blocks 16 --lr 0.0001 \
--label_in_use 0,3,4,10 --use_kpr --use_gn --vnt_not_use_ft --cross_p 1 --num_epochs 1500 --kpr_k 200
