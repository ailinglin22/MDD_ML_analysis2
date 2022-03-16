# MDDanalysis

## Data and Task:
The MDD data contains 13 symptom scores and 87 MRI image features. Totally, we have 85 subjects (50 Control subjects vs. 35 MDD subjects). Our task is try to select the certain biomarker as the reference so that we can conduct the clustering on the MDD subjcts for subgroups.

## Pipeline:
![](https://github.com/xtrigold/MDDanalysis/blob/main/images/pipeline.png)
Our data analysis pipeline is shown as the figure. We conduct the ML analysis on MDD data. Specifically, we firstly use the Canonical Correlation Analysis (CCA) on the data to select the biomarker, then we use the selected biomarker for the MDD subgroup clustering by hierarchical clustering. Finally, we conduct the RF classification on the two subclustering for feature ranking.

## Prerequisites
* [Python3](https://www.python.org/)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)
* [Sklearn](https://scikit-learn.org/stable/)
* [Pandas](https://pandas.pydata.org/docs/index.html)

We use the Python3.7. I upload the "myenv.yml", so the whole conda environment can be set by the "conda myenv create -f myenv.yml". More operations of conda can be found by the offical website. I also attach an toturial website [here](https://shandou.medium.com/export-and-create-conda-environment-with-yml-5de619fe5a2), you can find some others.
