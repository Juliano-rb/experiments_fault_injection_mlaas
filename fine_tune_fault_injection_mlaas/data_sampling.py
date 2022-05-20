from pandas import read_csv
from pandas.core.groupby.generic import DataFrameGroupBy, DataFrame

class DataSampling:
    def __init__(self, dataset_name):
        self.file_path = dataset_name 

    def get_by_width( \
            self, \
            number_of_instances: int, \
            min_width: int, \
            max_width: int, \
            enforce_balanced: bool = True \
        ) -> DataFrame:
        """
        Sample the dataset according to text width. \
        Tries to return a balanced dataset unless there are not suficient data.
        # Parameters
        - number_of_instances: int

            The final number of instances to be retrieved. If there are not
            suficient data for a balanced result, the dataset size can be lower than the desired.
        - min_width: int

            Defines the inferior limit of the width of the sampled sentences from the dataset
        - max_width: int

            Defines the superior limit of the width of the sampled sentences from the dataset
        - enforce_balanced: bool, default True
        
            When true, forces a balanced dataset result by reducing all classes to the one that has fewer instances.
        """
        dataset: DataFrame = read_csv(self.file_path)
        dataset = dataset[["airline_sentiment", "text"]]

        mask = (dataset["text"].str.len() >= min_width) & (dataset["text"].str.len() <= max_width)
        sub_set: DataFrame = dataset.loc[mask]

        # obtem uma quantidade balanceada de cada classe
        grouped: DataFrameGroupBy  = sub_set.groupby("airline_sentiment")
        number_of_classes = len(grouped)
        instances_of_each_class =  round(number_of_instances/number_of_classes)

        # check if force balanced when data in class is insuficient
        min_class_size = grouped.size().min()
        if enforce_balanced and instances_of_each_class > min_class_size:
            instances_of_each_class = min_class_size
        
        balanced = grouped.apply(lambda x: x.sample(instances_of_each_class if instances_of_each_class <= len(x) else min_class_size)).reset_index(drop=True)

        return balanced
    def get_by_word_count( \
            self, \
            number_of_instances: int, \
            min_count: int, \
            max_count: int, \
            enforce_balanced: bool = True \
        ) -> DataFrame:
        """
        Sample the dataset according to text width. \
        Tries to return a balanced dataset unless there are not suficient data.
        # Parameters
        - number_of_instances: int

            The final number of instances to be retrieved. If there are not
            suficient data for a balanced result, the dataset size can be lower than the desired.
        - min_width: int

            Defines the inferior limit of the width of the sampled sentences from the dataset
        - max_width: int

            Defines the superior limit of the width of the sampled sentences from the dataset
        - enforce_balanced: bool, default True
        
            When true, forces a balanced dataset result by reducing all classes to the one that has fewer instances.
        """
        dataset: DataFrame = read_csv(self.file_path)
        dataset = dataset[["airline_sentiment", "text"]]

        mask = (dataset["text"].str.split().apply(len) >= min_count) \
             & (dataset["text"].str.split().apply(len) <= max_count)
        sub_set: DataFrame = dataset.loc[mask]

        # obtem uma quantidade balanceada de cada classe
        grouped: DataFrameGroupBy  = sub_set.groupby("airline_sentiment")
        number_of_classes = len(grouped)
        instances_of_each_class =  round(number_of_instances/number_of_classes)

        # check if force balanced when data in class is insuficient
        min_class_size = grouped.size().min()
        if enforce_balanced and instances_of_each_class > min_class_size:
            instances_of_each_class = min_class_size
        
        balanced = grouped.apply(lambda x: x.sample(instances_of_each_class if instances_of_each_class <= len(x) else min_class_size)).reset_index(drop=True)

        return balanced
