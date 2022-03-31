from merlin.models.data.synthetic import SyntheticData

if __name__ == "__main__":
    for key in SyntheticData.DATASETS.keys():
        dataset = SyntheticData(key)
        dataset.generate_interactions(num_rows=1000, save=True)
