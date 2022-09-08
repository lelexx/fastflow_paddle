dir_name=exp
cd ./models
mkdir $dir_name
cd ..
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat toothbrush
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat screw
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat hazelnut
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat transistor
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat tile
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat pill
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat bottle
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat cable
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat capsule
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat carpet
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat grid
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat leather
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat metal_nut
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat wood
python train.py  -cfg ./configs/resnet18.yaml --data ./data --exp_dir $dir_name -cat zipper
python eval.py  -cfg ./configs/resnet18.yaml --data ./data -cat all --exp_dir $dir_name/
