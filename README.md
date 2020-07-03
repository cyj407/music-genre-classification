# Music Genre Classification
## About
- Use 5-fold validation and voting classifier to predict the music genre.
- Use timbral texture features and rhythmic content features to classify.
## Set Up
### Resource
- Download [res.zip](https://drive.google.com/file/d/13vY7X-zC2yKvWPliPE7WbQUF_FVfqO-n/view?usp=sharing), and unzip it.
- `res/` contains ten sorts of music genre. In each genre, there are 50 wav files.
- The files are selected from [Musical Genre Classification of Audio Signals](https://pdfs.semanticscholar.org/4ccb/0d37c69200dc63d1f757eafb36ef4853c178.pdf).

### Environment
- Python 3.6.5
- Sklearn 0.22.2.post1
- Pandas 1.0.3
- librosa 0.7.2

## Execute
```
python main.py
```
## Performance
- 80.6 % using voting classifier mixed with Random Forest, SVM and Logistic Regression.