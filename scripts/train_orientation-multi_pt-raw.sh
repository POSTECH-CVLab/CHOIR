python train.py -n rebuttal_pt-raw --num_gpus 2 --batch_size 128 --backbone identity --head pt --pt_num_blocks 4 --lr 0.0001 \
--label_in_use 0,3,4,10 --use_kpr --use_gn --vnt_not_use_ft --cross_p 1 --num_epochs 1500 --kpr_k 200
