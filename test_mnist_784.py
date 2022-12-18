# pylint: disable=import-error
""" mnist_784の精度をテストする """

# ライブラリのインポート
import mnist_784

cv_scores = mnist_784.cv_scores

# スコアを出力
print(sum(cv_scores) / len(cv_scores))

# 精度が0.85以下の場合、例外を発生させる
if sum(cv_scores) / len(cv_scores) <= 0.85:
    raise Exception('精度が0.85を下回っています')
