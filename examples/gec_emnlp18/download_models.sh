set -e
set -x

MODEL_DIR_URL=https://tinyurl.com/ybykuqap

mkdir -p models

# downloading predictors
mkdir -p models/cnn_predictor
mkdir -p models/rnn_predictor
curl -L -o models/cnn_predictor/model.best.pt $MODEL_DIR_URL/cnn_predictor/model.best.pt
curl -L -o models/rnn_predictor/model.best.pt $MODEL_DIR_URL/rnn_predictor/model.best.pt

# downloading HTER estimators
for emodel in RR CR CC RC; do
    curl -L o models/hter.$emodel.pt $MODEL_DIR_URL/hter.$emodel.pt
done

# downloading M2 estimators
for emodel in RR CR CC RC; do
    curl -L o models/m2scores.$emodel.pt $MODEL_DIR_URL/m2scores.$emodel.pt
done
