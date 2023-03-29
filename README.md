# 1. Download Model From Google Drive by using gdown

## best.pth
```
gdown https://drive.google.com/uc?id=146lzqc1Olga2pZyJK7dlyNL8EzQQA7sg
```

## worse.pth
```
gdown https://drive.google.com/uc?id=1gB_he0kYmOANxKL4rePkVjk1EkPbH9tX
```
# 2.download packages of requirements.txt
```
pip install -r requirements.txt
```
# 3.run the scripts x2_main.py

options: -m (model_name) -i (default='inputs','Input image or folder') -o (default='results', help='Output folder') -s (default=2, help='The final upsampling scale of the image')

## example:
```
python x2_main.py -m best.pth -o outputs -i inputs -s 2
```
```
python x2_main.py -m worse.pth -o outputs_2 -i inputs -s 2
```
And the results are in the outputs file
