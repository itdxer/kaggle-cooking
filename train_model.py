import os
import logging
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import ensemble, metrics

from src.utils import DATA_DIR, logtime, read_file_with_receipts


PATH_TO_TRAIN_FILE = os.path.join(DATA_DIR, 'train.json')
PATH_TO_TEST_FILE = os.path.join(DATA_DIR, 'test.json')
PATH_TO_SUBMISSION_FILE = os.path.join(DATA_DIR, 'submission.csv')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', dest='training_mode',
                    default='validation', help='set up training mode',
                    choices=['validation', 'submission'])
parser.add_argument('-r', '--reproducible', dest='is_reproducible',
                    action='store_true', help='make training reproducible')

classifier = Pipeline([
    ('tfidf_vectorizer', TfidfVectorizer(
        stop_words=['g', 'lb', 's', 'n'],
        lowercase=True,
    )),
    ('rf_classifier', ensemble.RandomForestClassifier(
        n_estimators=200,
        verbose=1,
        n_jobs=-1,
    )),
])

if __name__ == '__main__':
    args = parser.parse_args()

    logging.info("Start training")

    if args.is_reproducible:
        logging.info("Script runs in reproducible mode")
        np.random.seed(0)

    with logtime("Load train dataset"):
        train_data = read_file_with_receipts(PATH_TO_TRAIN_FILE)

    if args.training_mode == 'validation':
        with logtime("Train classifier"):
            x_train, x_test, y_train, y_test = train_test_split(
                train_data.ingredients,
                train_data.cuisine,
                train_size=0.8,
            )
            classifier.fit(x_train, y_train)

        with logtime("Check the perfomance"):
            y_predicted = classifier.predict(x_test)

            report = metrics.classification_report(y_test, y_predicted)
            logging.info("\n{}".format(report))

            score = metrics.accuracy_score(y_test, y_predicted)
            logging.info("Validation accuracy: {:.2f}%".format(100 * score))

    else:
        with logtime("Load test dataset"):
            test_data = read_file_with_receipts(PATH_TO_TEST_FILE)

        with logtime("Train classifier"):
            classifier.fit(train_data.ingredients, train_data.cuisine)

        with logtime("Classify test data"):
            test_data['cuisine'] = classifier.predict(test_data.ingredients)

        with logtime("Save test data in the CSV file"):
            test_data.to_csv(PATH_TO_SUBMISSION_FILE,
                             columns=['id', 'cuisine'], index=False)
            logging.info("The CSV file has been saved in the {} file"
                         "".format(PATH_TO_SUBMISSION_FILE))

    logging.info("Training finished")
