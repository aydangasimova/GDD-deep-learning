from sklearn.feature_extraction import text
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn import model_selection


pipeline = make_pipeline(text.TfidfVectorizer(), BernoulliNB())
model_selection.cross_val_score(
    pipeline, X_train_translated, y_train, cv=5, scoring="accuracy"
)
