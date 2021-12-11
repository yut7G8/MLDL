# Easy CNN

# フォルダの構成

```
src_ml_ict_cnn/
        │
        ├── main.py : 実行ファイル.これを実行し、model.pyのCNNクラスを呼び出す.
        ├── module/
        │       ├── activationfun.py : 活性化関数(シグモイド関数)を保存.
        │       └── model.py : CNNのモデルを保存.
        ├── result/
        │       ├── E1_transition.jpeg : 入力[1, 1, 0, 0], 目標出力[0, 1]の場合のEの減少をプロットしたもの.
        │       ├── E2_transition.jpeg : 入力[0, 0, 1, 1], 目標出力[1, 0]の場合のEの減少をプロットしたもの.
        │       └── E999_transition.jpeg : 入力[1, 1, 1, 1], 目標出力[1, 1]の場合のEの減少をプロットしたもの.
        └── README.md : 本ファイル.各種説明記載.

```

# 実行

```
$ python main.py

0.8950764525068631
0.8948086181767804
0.8945395933976503
0.8942693708297038
0.8939979430775449
...
0.035691085907107525
0.035611643140604804
0.035532530832671896
0.03545374706047679
0.03537528991525946
```

標準出力として、1000epoch分のEを出力する.