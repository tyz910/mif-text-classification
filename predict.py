import sys
from sklearn.externals import joblib

text = ' '.join(sys.argv[1:])
models = joblib.load('models.pkl')

X = models['transform']['words'].transform([text])
cats = models['category']['model'].predict(X)
tags = models['transform']['tags'].inverse_transform(models['tags']['model'].predict(X))

print('Направление: ' + cats[0])
print('Теги: ' + ', '.join(sorted(tags[0])))
