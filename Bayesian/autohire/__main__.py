import numpy as np

from autohire.utils import parse_pdf, parse_resume_df
from autohire.bow import BagOfWords
from autohire.encoder import LabelEncoder
from autohire.model import BayesianMulticlassModel

if __name__ == "__main__":
    x_train, y_train = parse_resume_df()
    bag_of_words = BagOfWords(x_train)
    label_encoder = LabelEncoder(y_train)

    x_train = bag_of_words.get_counts(x_train)
    y_train = label_encoder.encode(y_train)
    model = BayesianMulticlassModel(len(label_encoder), len(bag_of_words))
    model.fit(x_train=x_train, y_train=y_train)

    x_test = parse_pdf("data/resumes/computers_1.pdf")
    x_test = bag_of_words.get_counts(x_test)
    result = model.predict(x_test)
    result = label_encoder.decode(result)

    for job in result[:5]:
        print(job)
