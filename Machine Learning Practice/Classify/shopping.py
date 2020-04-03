import csv
import sys
import calendar
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")
    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence=[]
    labels=[]
    with open(filename, newline='') as csvfile:
         rows = csv.reader(csvfile)
         for i in rows:
            evidence.append(i[:-1])
            labels.append(i[-1])
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
    evidence=evidence[1:]
    labels=labels[1:]
    final=[]
    for i in evidence:
        new=[]
        for j in i:
            if re.match(r'^-?\d+(?:\.\d+)?$', j)!=None :
                new.append(float(j))
            elif len(j)==3 :
                j=abbr_to_num[j]-1
                new.append(j)
            elif j=="Returning_Visitor":
                new.append(1)
            elif j=="New_Visitor" or j=="Other":
                new.append(0)
            elif j=="FALSE":
                new.append(0)
            else:
                new.append(1)
        final.append(new)
    newy=[]
    for i in labels:
        if i=="FALSE":
            newy.append(0)
        else:
            newy.append(1)
     
    return (final,newy)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)
    return neigh

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    tpos=sum(np.array(labels)==1)
    tneg=sum(np.array(labels)==0)
    zero=0
    one=0
    for i in range(len(labels)):
        ans=labels[i]
        if ans==0 and predictions[i]==0:
            zero+=1
        if ans==1 and predictions[i]==1:
            one+=1
            
    sensitivity=one/tpos
    specificity=zero/tneg
    return sensitivity, specificity

if __name__ == "__main__":
    main()
