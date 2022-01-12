python scripts/train.py --n_layers=1 --n_heads=1 --d_model=4
python scripts/train.py --n_layers=1 --n_heads=4 --d_model=16 --load_path=./new_log/checkpoints/final_4_1_1.pt
python scripts/train.py --n_layers=1 --n_heads=4 --d_model=32 --load_path=./new_log/checkpoints/final_16_4_1.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=64 --load_path=./new_log/checkpoints/final_32_4_1.pt
python scripts/train.py --n_layers=2 --n_heads=4 --d_model=128 --load_path=./new_log/checkpoints/final_64_4_2.pt