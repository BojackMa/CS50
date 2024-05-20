import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# pip3 install scikit-learn
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
    返回地形式是([[],[],[],[],[]],[1,1,1,1,1,])，返回两个大列表，用元组括起来
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
    evidence = []
    labels = []
    # 下面是常见的处理手段，可以学习一下
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # Convert each column to the appropriate type
            row_evidence = [
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_to_index(row[10]),  # Month
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                visitor_type_to_int(row[15]),  # VisitorType
                1 if row[16].lower() == 'true' else 0  # Weekend
            ]
            evidence.append(row_evidence)
            labels.append(1 if row[17].lower() == 'true' else 0)  # Revenue
    return (evidence, labels)

# 下面的处理方式我学习到了，利用列表
def month_to_index(month):
    """Convert month from name to index (0 for January, 11 for December)."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months.index(month)

def visitor_type_to_int(visitor_type):
    """Convert visitor type to integer (0 for not returning, 1 for returning)."""
    return 1 if visitor_type == "Returning_Visitor" else 0

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # 创建一个 KNN 分类器实例，其中 n_neighbors=1 指定 K=1
    model = KNeighborsClassifier(n_neighbors=1)

    # 使用提供的证据和标签训练模型
    model.fit(evidence, labels)

    # 返回训练好的模型
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Sensitivity 衡量的是模型正确识别正类（如疾病阳性、有欺诈行为等）的能力。它的定义是所有实际正类中，被模型正确预测为正类的比例。
    # Specificity 衡量的是模型正确识别负类（如疾病阴性、无欺诈行为等）的能力。它的定义是所有实际负类中，被模型正确预测为负类的比例。
    # 在实际应用中，通常需要在Sensitivity和Specificity之间找到一个平衡。例如，在癌症筛查中，你可能希望具有高敏感性（即尽可能少地漏检病例），即使这意味着一定程度上的误报增加（降低特异性）。而在一些需要高度精确的场景（如选择手术病人），则可能更偏好高特异性。
    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if predicted == 1:
                true_positive += 1
        elif actual == 0:
            total_negative += 1
            if predicted == 0:
                true_negative += 1
    # 下面写if条件句是为了防止分母为0，但实际上不会有这种情况
    sensitivity = true_positive / total_positive if total_positive else 0
    specificity = true_negative / total_negative if total_negative else 0
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
