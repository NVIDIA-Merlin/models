import numpy as np
import nvtabular as nvt


def workflow_fit_transform(outputs, train_path, valid_path, output_path, workflow_name=None):
    """fits and transforms datasets applying NVT workflow"""
    workflow = nvt.Workflow(outputs)

    train_dataset = nvt.Dataset(train_path)
    valid_dataset = nvt.Dataset(valid_path)

    workflow.fit(train_dataset)

    workflow.transform(train_dataset).to_parquet(output_path=output_path + "/train/")

    workflow.transform(valid_dataset).to_parquet(output_path=output_path + "/valid/")

    if workflow_name is None:
        workflow_name = "workflow"
    else:
        workflow_name = workflow_name
    # save workflow to the pwd
    workflow.save(workflow_name)


def save_results(model_name, model):
    """a funct to save valudation accurracy results in a text file"""
    with open("results.txt", "a") as f:
        f.write(model_name)
        f.write("\n")
        for key, value in model.history.history.items():
            if "val_auc" in key:
                f.write("%s:%s\n" % (key, value[0]))


def create_bar_chart(text_file_name, models_name):
    import matplotlib.pyplot as plt

    """a func to plot barcharts via parsing the  accurracy results in a text file"""
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
