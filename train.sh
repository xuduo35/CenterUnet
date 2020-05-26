nohup python3 -u main.py --network_type unetobj --backbone peleenet --batch_size 16 --train_phase pre_train_center --lr 0.001 $* >./logs/centernet.log 2>&1 &
