import nvtabular as nvt


def workflow_fit_transform(outputs, train_path, test_path, output_path):
    workflow = nvt.Workflow(outputs)

    train_dataset = nvt.Dataset(train_path)
    test_dataset = nvt.Dataset(test_path)

    workflow.fit(train_dataset)

    workflow.transform(train_dataset).to_parquet(output_path=output_path + "/train/")

    workflow.transform(test_dataset).to_parquet(output_path=output_path + "/test/")
