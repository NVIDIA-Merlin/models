import merlin.models.tf as ml

if __name__ == "__main__":
    for path in ml.SyntheticData.DATASETS.values():
        schema = ml.SyntheticData.read_schema(path)
        ml.SyntheticData.from_schema(schema, output_dir=path, num_rows=1000)
