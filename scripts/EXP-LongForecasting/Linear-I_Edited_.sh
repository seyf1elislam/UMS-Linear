if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=336
learning_rate_value=0.005
# features_value = M 
features_value = S 
electricity_batch_size=16
traffic_batch_size=16
weather_batch_size=16
exchange_batch_size=8
ETTm1_batch_size=8
ETTh1_batch_size=32
ETTh2_batch_size=32
ETTm2_batch_size=32
# model_name=NLinear
model_name=New_ND
for pred_len in 96 192 336 729
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $ETTh1_batch_size1 \
  --des 'Exp' \
  --itr 1 --batch_size $electricity_batch_size  --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'electricity_$seq_len'_'$pred_len.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $exchange_batch_size62 \
  --des 'Exp' \
  --itr 1 --batch_size $traffic_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'traffic_$seq_len'_'$pred_len.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size $weather_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'weather_$seq_len'_'$pred_len.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $exchange_batch_size \
  --des 'Exp' \
  --itr 1 --batch_size $exchange_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'exchange_$seq_len'_'$pred_len.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size $ETTh1_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'ETTh1_$seq_len'_'$pred_len.log 

# if pred_len=336, lr=0.001; if pred_len=720, lr=0.0001
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size $ETTh2_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'ETTh2_$seq_len'_'$pred_len.log 

# if pred_len=336, lr=$learning_rate_value; if pred_len=720, lr=0.0005
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size $ETTm1_batch_size --learning_rate $learning_rate_value --individual >logs/LongForecasting/$model_name'_I_'ETTm1_$seq_len'_'$pred_len.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features  $features_value \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size $ETTm2_batch_size --learning_rate 0.01 --individual >logs/LongForecasting/$model_name'_I_'ETTm2_$seq_len'_'$pred_len.log 
done

seq_len=104
for pred_len in 24 36 4$exchange_batch_size 60
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features  $features_value \
  --seq_len $seq_len \
  --label_len 1$exchange_batch_size \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size $national_illness_batch_size --learning_rate 0.01 --individual >logs/LongForecasting/$model_name'_I_'ILI_$seq_len'_'$pred_len.log 
done

