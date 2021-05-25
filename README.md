# Comparison of Machine Learning Techniques to Handle Imbalanced COVID-19 CBC Datasets

Welcome!

The Coronavirus pandemic caused by the novel SARS-CoV-2 has significantly impacted human health and the economy, especially in countries struggling with financial resources for medical testing and treatment, such as Brazil's case, the third most affected country by the pandemic. In this scenario, machine learning techniques have been heavily employed to analyze different types of medical data, and aid decision making, offering a low-cost alternative. Due to the urgency to fight the pandemic, a massive amount of works are applying machine learning approaches to clinical data, including complete blood count (CBC) tests, which are among the most widely available medical tests. In this work, we review the most employed machine learning classifiers for CBC data, together with popular sampling methods to deal with the class imbalance. Additionally, we describe and critically analyze three publicly available Brazilian COVID-19 CBC datasets and evaluate the performance of eight classifiers and five sampling techniques on the selected datasets. Our work provides a panorama of which classifier and sampling methods provide the best results for different relevant metrics and discuss their impact on future analyses. The metrics and algorithms are introduced in a way to aid newcomers to the field. Finally, the panorama discussed here can significantly benefit the comparison of the results of new ML algorithms.

## Datasets

Complete datasets used in the present study were obtained from an open repository of COVID-19-related cases in Brazil. The database is part of the _COVID-19 Data Sharing/BR_ initiative, and it is comprised of information about approximately 177,000 clinical cases. Patient data were collected from February 26th to June 30th, 2020 from three distinct private health services providers in the São Paulo State, namely:

- Fleury Group: [DATA](FLEURY/DATA/) - [https://www.fleury.com.br](https://www.fleury.com.br)
- Albert Einstein Hospital: [DATA](AE/DATA/) - [https://www.einstein.br](https://www.einstein.br)
- Sírio-Libanês Hospital: [DATA](HSL/DATA/) - [https://www.hospitalsiriolibanes.org.br](https://www.hospitalsiriolibanes.org.br)

The classes are listed in the "y" column in the .csv files. Values of 0 indicate negative RT-PCR results and values of 1 indicate positive RT-PCR results. For more information regarding the datasets please refer to the main publication.

## Methods

The following algorithms, methods, and tools were used in our experiments. For complete details about each one please refer to the main publication.

- Classifiers: Logistic Regression, SVM, MLP, Naive Bayes, Random Forest, Decision Tree, KNN, XGBC
- Sampling algorithms: ROS, RUS, ADASYN, SMOTE, SMOTE-Tomek
- Metrics: Sensitivity, Specificity, LR+, LR-, DOR, F1-score
- Data characterization: Kolmogorov-Smirnov test, Bhattacharyya Distance, violin plot

## Libraries

This implementation of relevance aggregation uses the following [Python 3.7](https://www.python.org/) libraries:

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
- [imbalanced-learn](https://imbalanced-learn.org/stable/)

## How to cite

If you use our code, methods, or results in your research, please cite the main publication:

- Marcio Dorn et al. _Comparison of Machine Learning Techniques to Handle Imbalanced COVID-19 CBC Datasets_, PeerJ Computer Science, Volume X, Month **Year**, Pages x-x, DOI: [xxxx]()

Bibtex entry:
```
@article{,
  title={},
  author={},
  journal={},
  year={},
  doi = {},
  publisher={}
}
```

## Contact information

- [Dr. Marcio Dorn](https://orcid.org/0000-0001-8534-3480) - Associate Professor ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

    - mdorn@inf.ufrgs.br

- [Bruno I. Grisci](https://orcid.org/0000-0003-4083-5881) - PhD student ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

- [Pedro Narloch](https://scholar.google.com/citations?user=KFfidFEAAAAJ&hl=pt-PT) - PhD student ([Institute of Informatics](https://www.inf.ufrgs.br/site/en) - [UFRGS](http://www.ufrgs.br/english/home))

- [Dr. Bruno César Feltes](https://orcid.org/0000-0002-2825-8295) - Post-Doctorate ([Center for Biotechnology](https://www.ufrgs.br/ppgbcm/?lang=en) - [UFRGS](http://www.ufrgs.br/english/home))

- [Dr. Eduardo Avila]() - Post-Doctorate ([Forensic Genetics Laboratory, School of Health and Life Sciences](https://www.pucrs.br/health-sciences/) - [PUCRS](https://www.pucrs.br/en/))

- [Dr. Alessandro Kahmann](http://lattes.cnpq.br/4661839485236719) - Professor ([Institute of Mathematics, Statistics and Physics](https://imef.furg.br/) - [FURG](https://www.furg.br/en/))

- [Dr. Clarice Alho](https://orcid.org/0000-0002-4819-9587) - Professor ([Forensic Genetics Laboratory, School of Health and Life Sciences](https://www.pucrs.br/health-sciences/) - [PUCRS](https://www.pucrs.br/en/))

- Structural Bioinformatics and Computational Biology Lab: [http://sbcb.inf.ufrgs.br/](http://sbcb.inf.ufrgs.br/)