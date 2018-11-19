
# NeuQE (Neural Quality Estimation)

Neural quality estimation toolkit which can be used for natural language generation tasks such as grammatical error correction, machine translation, simplification, and summarization.

The source code is repository was used in this paper: "Neural Quality Estimation Models for Grammatical Error Correction" (EMNLP 2018).

If you use this code for your work, please cite this [paper](http://aclweb.org/anthology/D18-1274):
```
@InProceedings{chollampatt2018neuqe,
    title = {Neural Quality Estimation of Grammatical Error Correction},
    authors = {Chollampatt, Shamil and Ng, Hwee Tou},
    booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    month     = {November},
    year = {2018},
    address = {Brussels, Belgium}
}
```

## Prerequisites
Python 3
PyTorch 0.3

## Training


### Pre-training the predictor model

 To train the predictor models, use the script `train_predictor.py`. See the available options by using `--help` flag


For example, the CNN-based predictor for EMNLP-2018 GEC QE was trained using the following command:
```
 python train_predictor.py -train $TRAIN_PATH_PREFIX -valid $VALID_PATH_PREFIX -ssuf src -tsuf trg \
    -arch cnn -nsvocab 30000  -ntvocab 30000 \
    -nslayers 7 -ntlayers 7 -skwidth 3 -tkwidth 3 \
    -nhid 700 -nsembed 500 -ntembed 500 \
    -nepochs 10 -bsize 64 -lrate 1.0 -cnorm 5.0 -maxslen 50 -maxtlen 50 \
    -logafter 1000 -outdir $MODEL_OUT_DIR
```

For training the RNN-based predictor, the following command was used:
```
python train_predictor.py -train $TRAIN_PATH_PREFIX -valid $VALID_PATH_PREFIX -ssuf src -tsuf trg \
    -arch rnn -nsvocab 30000 -ntvocab 30000 \
    -nhid 700 -nsembed 500 -ntembed 500 \
    -nepochs 10 -bsize 64 -lrate 1.0 -cnorm 5.0 -maxslen 50 -maxtlen 50 \
    -logafter 1000 -outdir $MODEL_OUT_DIR

```

### Training the estimator model

To train the estimator model, use the script `train_estimator.py`. See the available options by using the `--help` flag.

For training the estimator for EMNLP-2018 GEC QE model, the following command was used (`$ARCH` can be `cnn` or `rnn`):
```
python train_estimator.py \
    -train $QE_TRAIN_DATA_PATH_PREFIX \
    -valid $QE_VALID_DATA_PATH_PREFIX \
    -ssuf src -hsuf hyp -scoresuf $SCORE_SUFFIX \
    -pmodel $PRED_MODEL_PATH \
    -arch $ARCH -nhid 100 -qvectype pre  \
    -opt adam -lrate 0.0005 -bsize 32 -validbsize 1 -do 0.5 -nepochs 50 \
    -metrics pc mae rmse -outdir $EST_MODEL_OUT_DIR
```
The GEC system used for generating system hypotheses for training the EMNLP-2018 GEC QE system was a multilayer convolutional sequence-to-sequence model trained on Lang-8 ([code](https://github.com/nusnlp/mlconvgec2018)). The downstream GEC system that was improved using QE scores also used the same underlying architecture with additional techniques described in the [paper](http://aclweb.org/anthology/D18-1274).

## Testing

To test the estimator model, use the script `test_predictor_estimator`. An example is shown below:

```
python test_predictor_estimator.py \
    -test $QE_TEST_DATA_PATH_PREFIX \
    -ssuf src -hsuf hyp -scoresuf $SCORE_SUFFIX \
    -pemodel $PRED_MODEL_PATH $EST_MODEL_PATH -metrics pc rmse -outdir $OUT_DIR
```
If you want to use multiple estimators while testing, use multiple `-pemodel` flags specifying the paths to each predictor-estimator model pair.

## Pre-trained Models

For downloading the pre-trained models used for quality estimation of grammatical error correction for EMNLP 2018 paper, run the `download_models.sh` script inside `examples/gec_emnlp18/` directory.

## License

The source code is licensed under GNU GPL 3.0 (see [LICENSE](LICENSE.md)) for non-commerical use. For commercial use of this code, separate commercial licensing is also available. Please contact:

* Shamil Chollampatt (shamil@u.nus.edu)
* Hwee Tou Ng (nght@comp.nus.edu.sg)
