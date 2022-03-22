import matplotlib.pyplot as plt
import numpy as np
import nvtabular as nvt


def workflow_fit_transform(outputs, train_path, test_path, output_path, workflow_name):
    workflow = nvt.Workflow(outputs)

    train_dataset = nvt.Dataset(train_path)
    test_dataset = nvt.Dataset(test_path)

    workflow.fit(train_dataset)

    workflow.transform(train_dataset).to_parquet(output_path=output_path + "/train/")

    workflow.transform(test_dataset).to_parquet(output_path=output_path + "/test/")
    if workflow_name is None:
        workflow_name = "workflow"
    else:
        workflow.save(workflow_name)


def save_results(model_name, model):
    with open("results.txt", "a") as f:
        f.write(model_name)
        f.write("\n")
        for key, value in model.history.history.items():
            if "val_auc" in key:
                f.write("%s:%s\n" % (key, value[0]))


def create_bar_chart(text_file_name, models_name):
    auc = []
    with open(text_file_name, "r") as infile:
        for line in infile:
            if "auc" in line:
                data = [line.rstrip().split(":")]
                key, value = zip(*data)
                auc.append(float(value[0]))

    X_axis = np.arange(len(models_name))

    plt.title("Models' accuracy metrics comparison", pad=20)
    plt.bar(X_axis - 0.2, auc, 0.4, label="AUC")

    plt.xticks(X_axis, models_name)
    plt.xlabel("Models")
    plt.ylabel("AUC")
    plt.show()
