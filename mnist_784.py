# pylint: disable=import-error
""" mnist_784のデータに対して、学習・推論する """

# ライブラリのインポート
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# MNISTデータのダウンロード
mnist_784 = fetch_openml('mnist_784', as_frame=False)


# 教師データ、テストデータを抽出する
# 実行するたびに結果が変わらないようにseedを固定する。
SEED = 0

# 「とりあえず」の評価なので、教師データは1000個、テストデータは300個とし、ラベルにより層化抽出する。)
X_train, X_test, y_train, y_test = train_test_split(
    mnist_784.data,
    mnist_784.target,
    test_size=300,
    train_size=1000,
    random_state=SEED,
    stratify=mnist_784.target)

# 0から255までの値なので255で割って規格化する。
X_train = X_train / 255
X_test = X_test / 255


# # 分類器: サポートベクターマシン(カーネルはRBF)
# classifier = SVC(gamma="auto", random_state=SEED)
# ランダムフォレスト  (木の数は100)  
classifier = RandomForestClassifier(random_state=SEED)


# 交差検証(10分割）を実行
cv_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)


# スコアを出力
print(f"score: {sum(cv_scores) / len(cv_scores)}")
