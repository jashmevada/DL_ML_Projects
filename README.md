# Deep Learning and Machine Learning Projects

This repository contains six Kaggle project notebooks: two deep-learning
projects and four classical machine-learning projects. The collection covers
data exploration, preprocessing, model comparison/training, and submission
creation.

The notebooks and their saved outputs are preserved as project artifacts. The
documentation below describes the code and configuration currently stored in
the notebooks; the notebooks were not re-executed while preparing this README.

## Project index

| Project | Notebook | Task | Target(s) | Final model used for submission |
| --- | --- | --- | --- | --- |
| Face age and gender prediction | [`face_age_gender_multitask_resnet.ipynb`](face_age_gender_multitask_resnet.ipynb) | Multitask image classification/regression | `gender`, `age` | ImageNet-pretrained ResNet-50 with separate heads |
| Protein secondary-structure prediction | [`protein_secondary_structure_cnn_bilstm.ipynb`](protein_secondary_structure_cnn_bilstm.ipynb) | Per-residue sequence labeling | `sst3`, `sst8` | Multi-kernel CNN + bidirectional LSTM |
| Session purchase-value prediction | [`session_purchase_value_two_stage_model.ipynb`](session_purchase_value_two_stage_model.ipynb) | Zero-inflated regression | `purchaseValue` | Preserved-output winner: tuned XGBoost classifier + tuned XGBoost regressor |
| Restaurant review rating prediction | [`restaurant_review_rating_linear_svc.ipynb`](restaurant_review_rating_linear_svc.ipynb) | Five-class text/tabular classification | `rating` | Tuned linear SVM (`LinearSVC`) |
| Bank customer churn prediction | [`bank_customer_churn_lightgbm.ipynb`](bank_customer_churn_lightgbm.ipynb) | Binary classification | `exit_status` | Tuned LightGBM classifier |
| Flight ticket price prediction | [`flight_price_random_forest.ipynb`](flight_price_random_forest.ipynb) | Tabular regression | `price` | Tuned random-forest regressor |

## Projects

### 1. Face age and gender prediction

**Notebook:**
[`face_age_gender_multitask_resnet.ipynb`](face_age_gender_multitask_resnet.ipynb)

- **Objective:** predict a binary gender label and a numerical age from each
  face image with one shared visual backbone and two output heads.
- **Dataset:** Kaggle competition source ID `119874`, bundle version
  `14372465`. The notebook reads
  `/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/train.csv` and
  `/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/test.csv`; images are
  resolved below `/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/` from the
  CSV column `full_path`.
- **Target and features:** `gender` is the binary classification target and
  `age` is the regression target. The model input is the RGB image referenced
  by `full_path`; `id` is retained for submission.
- **Preprocessing:** images are converted to RGB, resized, converted to tensors,
  and normalized with ImageNet means and standard deviations. The scratch CNN
  uses `128 x 128`; the ResNet path uses `224 x 224`. Training data is split
  randomly into 80% training and 20% validation with no stored split seed.
- **Models considered:** a three-block convolutional network trained from
  scratch and an ImageNet-pretrained ResNet-50 backbone. Both have separate
  linear gender and age heads and optimize binary cross-entropy plus age MSE.
- **Final selected path:** the stored configuration is `MODEL_TYPE = 1`,
  `TRAIN_MODE = 0`, `INF_MODE = 1`, and `LOAD_MODEL = 1`. It therefore uses the
  ResNet-50 model for inference and downloads `model_t1.pth` from KaggleHub
  handle `jashmevada/dlgenai-nppe1/pytorch/fine_tune`.
- **Evaluation:** validation logging includes gender BCE loss, age MSE loss,
  binary gender F1, age MAE/RMSE, a clipped age score based on RMSE, and a
  harmonic quantity monitored by early stopping/checkpointing. The README does
  not assert a score; the notebook has no saved training-result outputs.
- **Submission:** `submission.csv` has columns `id`, `gender`, and `age`.
  Gender probabilities are thresholded at `0.5`; ages are rounded to integers.
- **Execution caveats:** pretrained ResNet weights and the KaggleHub state
  dictionary must be downloadable. `HF_TOKEN` is read unconditionally. The
  validation function named `calculate_age_nrmse` actually returns
  `1 - min(RMSE, 30) / 30`, and the harmonic calculation subtracts that value
  from one again; interpret the logged metric according to the code rather than
  its name. Training saves the current in-memory model, not explicitly the
  best checkpoint selected by the callback.

### 2. Protein secondary-structure prediction

**Notebook:**
[`protein_secondary_structure_cnn_bilstm.ipynb`](protein_secondary_structure_cnn_bilstm.ipynb)

- **Objective:** label every amino-acid residue with both three-state (`sst3`)
  and eight-state (`sst8`) secondary structure.
- **Dataset:** Kaggle competition source ID `125543`, bundle version
  `14858151`. Inputs are
  `/kaggle/input/sep-25-dl-gen-ai-nppe-2/train.csv` and
  `/kaggle/input/sep-25-dl-gen-ai-nppe-2/test.csv`.
- **Target and features:** `seq` is the amino-acid sequence feature; `sst3` and
  `sst8` are equal-length per-residue target strings; `id` is used for
  submission.
- **Preprocessing:** the 20 standard amino acids are integer-encoded, `*` has a
  masked/unknown token, and zero is padding. Q3 uses `H/E/C`; Q8 uses
  `H/G/I/E/B/T/S/C`. Sequences and labels are padded or truncated to the
  dataset-wide maximum sequence length plus 10. The training/validation split
  is 85%/15%, seeded with `42`, and stratified by quantile bins of sequence
  length (with a fixed-width-bin fallback).
- **Models considered:** bidirectional vanilla RNN, bidirectional LSTM, and a
  CNN-BiLSTM. The CNN-BiLSTM concatenates kernel sizes 1, 3, 5, and 7, passes
  them through a bidirectional LSTM, predicts Q3, and supplies detached Q3
  probabilities as features for the Q8 head.
- **Final selected path:** `MODEL_TYPE = 2`, so the multi-kernel CNN-BiLSTM is
  used. The stored configuration has both training and inference enabled,
  `EPOCHS = 100`, and `LOAD_MODEL = 0`; inference consequently uses the model
  trained in the same session rather than downloading a state dictionary.
- **Evaluation:** padding-aware cross-entropy is summed across Q3 and Q8.
  Validation records padding-ignored macro F1 for each target and their
  harmonic mean (`val_score`). That score drives early stopping and the
  reduce-on-plateau learning-rate scheduler. No new result is claimed here.
- **Submission:** `submission.csv` has `id`, `sst8`, and `sst3`; predictions are
  decoded to strings and truncated to each original sequence length.
- **Execution caveats:** the saved output reports a discovered padded length of
  1,642, so batch size 32 can require substantial GPU memory. With the current
  flags, a top-to-bottom run trains, attempts to upload `model_t1.pth` to
  `jashmevada/dlgenai-nppe2/pytorch/cnn_lstm`, and then predicts. This requires
  `HF_TOKEN`, Trackio/Hugging Face access, and KaggleHub upload permission. The
  preserved output says a prior trainer stopped at two epochs and contains an
  older Trackio run name, while the stored source now requests 100 epochs;
  saved outputs should not be treated as results from the current settings.

### 3. Session purchase-value prediction

**Notebook:**
[`session_purchase_value_two_stage_model.ipynb`](session_purchase_value_two_stage_model.ipynb)

- **Objective:** predict session purchase value in a strongly zero-inflated
  web-analytics dataset by first predicting whether a purchase occurs and then
  estimating its amount.
- **Dataset:** Kaggle competition source ID `99546`, bundle version `11895149`,
  with
  `/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv` and
  `/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv`.
- **Target and features:** `purchaseValue` is the regression target. Inputs are
  the remaining user, session, device, browser, traffic-source, geography, and
  engagement fields. The notebook additionally creates
  `pageviewsPerSession` and per-user `userSessionCount`.
- **Preprocessing:** the code drops constant string columns containing only
  `not available in demo dataset`, columns over 80% missing, and
  `totals.visits`, `totals.bounces`, and `locationZone`. It makes an 80%/20%
  split seeded with `42` and stratified by purchase versus no purchase.
  Numeric values receive median imputation and standard scaling; categorical
  fields with at most 10 values receive most-frequent imputation and one-hot
  encoding; higher-cardinality fields use target encoding. Positive purchase
  amounts are trained in `log1p` space.
- **Models considered:** logistic/linear baselines, random forests, gradient
  boosting, histogram gradient boosting, XGBoost, LightGBM classification, and
  soft-voting ensembles. Classifiers and regressors are tuned separately with
  three-fold grid or randomized search.
- **Final selected path:** the preserved output identifies a tuned XGBoost
  classifier plus tuned XGBoost regressor as the winner. The code does not
  hard-code that outcome: it selects the tuned classifier with the highest
  holdout F1, selects the individual regressor with the lowest end-to-end
  holdout RMSE alongside that classifier, and then selects the voting ensemble
  instead if its end-to-end holdout RMSE is lower. Fresh runs may therefore
  choose a different classifier/regressor or the ensemble.
- **Evaluation:** the purchase classifier uses accuracy, precision, recall, F1,
  and three-fold CV F1. Regressors are tuned using negative RMSE in transformed
  target space. End-to-end predictions are evaluated on the original target
  scale with RMSE, MAE, R-squared, zero-prediction accuracy, and non-zero-only
  RMSE/MAE, then compared with mean, zero, and randomized median baselines. No
  undocumented score is claimed here.
- **Submission:** `submission.csv` has `id` and `purchaseValue`. The `id` is a
  zero-based range generated in the notebook; predicted amounts are clipped at
  zero after reversing the log transform.
- **Execution caveats:** the searches include hundreds of fits and can take a
  long time on CPU. The classifier's high-cardinality target encoder is fitted
  with raw `purchaseValue`, not the derived binary label. CV RMSE values in the
  regression-tuning section are measured in log-target space even though some
  printed labels include a currency symbol. One baseline uses unseeded random
  sampling, so that baseline is not reproducible across fresh runs.

### 4. Restaurant review rating prediction

**Notebook:**
[`restaurant_review_rating_linear_svc.ipynb`](restaurant_review_rating_linear_svc.ipynb)

- **Objective:** predict a restaurant review's integer star rating from review
  text and accompanying location/review metadata.
- **Dataset:** Kaggle competition source ID `111489`, bundle version
  `13272433`, slug `mlp-term-2-2025-kaggle-assignment-3`. The notebook reads
  `/kaggle/input/mlp-term-2-2025-kaggle-assignment-3/train.csv` and `test.csv`;
  the same directory also contains `sample_submission.csv`.
- **Target and features:** `rating` is a five-class target. Retained features
  are `latitude`, `longitude`, integer-converted `rating_count`, `review_time`,
  `review`, and engineered `review_length`. `id`, `store_name`, `category`, and
  `store_address` are removed from model input; test `id` is preserved for the
  submission.
- **Preprocessing:** an 80%/20% split uses seed `42`. Numeric features receive
  median imputation and scaling. Object columns receive most-frequent
  imputation and one-hot encoding. Review text is also lowercased, stripped of
  non-letter characters, and represented by TF-IDF with at most 800 features.
- **Models considered:** logistic regression, ridge classifier, decision tree,
  random forest, linear SVM, LightGBM, and Extra Trees. LightGBM, LinearSVC, and
  Extra Trees are subsequently tuned with five-fold stratified randomized
  search.
- **Final selected path:** the tuned `LinearSVC` stored in
  `best_models['LinearSVC']` creates the test predictions.
- **Evaluation:** the holdout comparison reports accuracy and weighted
  precision, recall, and F1. Initial models also use five-fold stratified CV;
  the actual CV scorer is accuracy. Hyperparameter search is likewise scored by
  accuracy. No score beyond the preserved notebook output is asserted here.
- **Submission:** `submission.csv` has `id` and integer `rating`.
- **Execution caveats:** `review` remains in the general categorical-column
  list while also being passed to TF-IDF, so it is represented twice and its
  one-hot branch can be very wide. The result columns named `CV F1 Mean/Std`
  actually contain accuracy CV scores. Saved output includes LinearSVC and
  logistic convergence warnings, and the ridge model fails under the recorded
  environment because of a SciPy/scikit-learn solver API mismatch.

### 5. Bank customer churn prediction

**Notebook:**
[`bank_customer_churn_lightgbm.ipynb`](bank_customer_churn_lightgbm.ipynb)

- **Objective:** predict whether a bank customer exits.
- **Dataset:** Kaggle competition source ID `107399`, bundle version
  `13009703`, slug `mlp-term-2-2025-kaggle-assignment-2`. The notebook reads
  `/kaggle/input/mlp-term-2-2025-kaggle-assignment-2/train.csv` and `test.csv`;
  `sample_submission.csv` is present in the same attachment.
- **Target and features:** binary target `exit_status`. Numeric features are
  `credit_score`, `age`, `tenure`, `acc_balance`, `prod_count`, `has_card`,
  `is_active`, and `estimated_salary`; categorical features retained are
  `country` and `gender`. `id`, `customer_id`, and high-cardinality `last_name`
  are excluded from model input.
- **Preprocessing:** an 80%/20% split uses seed `42`. Numeric fields receive
  mean imputation and standard scaling. Missing countries become `unknown`;
  country and gender are one-hot encoded with the first level dropped and
  unknown categories ignored.
- **Models considered:** logistic and ridge classifiers, decision tree, random
  forest, gradient boosting, histogram gradient boosting, XGBoost, LightGBM,
  and k-nearest neighbors. LightGBM, histogram gradient boosting, and XGBoost
  are then tuned with five-fold stratified randomized search scored by F1.
- **Final selected path:** tuned LightGBM from `best_models['LightGBM']`.
- **Evaluation:** holdout accuracy, precision, recall, F1, and ROC AUC where
  probabilities are available, plus five-fold stratified CV F1 and tuning-time
  F1 comparison. The README does not manufacture or extrapolate results.
- **Submission:** `submission.csv` has `id` and binary `exit_status`.
- **Execution caveats:** the initial holdout split is not stratified even though
  later CV is. The source code uses mean numeric imputation, while its printed
  explanatory text says median. Use the implementation, not that stale label,
  when reproducing preprocessing.

### 6. Flight ticket price prediction

**Notebook:**
[`flight_price_random_forest.ipynb`](flight_price_random_forest.ipynb)

- **Objective:** estimate flight ticket prices from route, schedule, carrier,
  class, trip-duration, and booking-lead-time information.
- **Dataset:** Kaggle competition source ID `105665`, bundle version
  `12770423`, slug `mlp-term-2-2025-kaggle-assignment-1`. The notebook reads
  `/kaggle/input/mlp-term-2-2025-kaggle-assignment-1/train.csv` and `test.csv`;
  `sample_submission.csv` is in the same directory.
- **Target and features:** `price` is the regression target. Features are
  `airline`, `flight`, `source`, `departure`, `stops`, `arrival`,
  `destination`, `class`, `duration`, and `days_left`; `id` is submission-only.
- **Preprocessing:** training rows missing selected categorical fields are
  dropped. Sequential IQR filtering is applied to `price`, `duration`, and
  `days_left`; because the filter comparisons exclude nulls, it also removes
  rows with missing values in those columns. The remaining data is split
  80%/20% with seed `42`. Categorical columns with more than 15 values use
  target encoding; lower-cardinality columns use most-frequent imputation and
  one-hot encoding. Numeric columns use mean imputation and standard scaling.
- **Models considered:** linear regression, ridge, lasso, random forest, KNN,
  gradient boosting, and XGBoost. Nontrivial models use three-fold grid search
  scored by negative root mean squared error.
- **Final selected path:** the tuned random forest from
  `best_models['Random Forest']`, selected by the notebook's holdout comparison.
- **Evaluation:** holdout R-squared, adjusted R-squared, RMSE, MAE, and MAPE,
  with visual comparisons. No new performance value is claimed here.
- **Submission:** `submission.csv` has `id` and predicted `price` for the 10,000
  test rows.
- **Execution caveats:** target-based `price` outlier filtering occurs before
  the train/validation split, so the validation population has already been
  filtered using the target. The first model-comparison cell stores
  `mean_squared_error` in a column called `RMSE` without taking a square root;
  the later detailed metrics cell computes RMSE correctly. Training drops
  missing rows, while test data relies on the fitted imputers, so the train and
  inference missing-data policies differ.

## Reproducibility notes

- Saved cell outputs document historical runs, not guaranteed fresh results
  from the current source. This is explicit in the protein notebook, whose
  output and current epoch/run-name settings differ.
- Random seeds are used in most classical-ML splits and estimators, but not in
  every operation. The face train/validation split and the purchase notebook's
  randomized median baseline are unseeded.
- The notebooks write only a generic `submission.csv`; rename or move each
  output before running another notebook.
- No trained model artifacts or competition datasets are stored in this
  repository. They must be obtained from Kaggle/KaggleHub or regenerated.
