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
