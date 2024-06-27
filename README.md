# netfeat
---
This repository contains the code and supporting documents associated with the following manuscript:

Please cite as:


---
## Authors:
* Juliana Gonzalez-Astudillo, Postdoctoral Researcher, Nerv team-project, Inria Paris, Paris Brain Institute
* [Fabrizio De Vico Fallani](https://sites.google.com/site/devicofallanifabrizio/), Research Scientist, Nerv team-project, Inria Paris, Paris Brain Institute


---
## Abstract
Brain-computer interfaces (BCIs) enable users to interact with the external world using brain activity. 
Despite their potential in neuroscience and industry, BCI performance remains inconsistent in noninvasive applications, often prioritizing algorithms that achieve high classification accuracies while masking the neural mechanisms driving that performance. 
In this study, we investigated the interpretability of features derived from brain network lateralization, benchmarking against widely used techniques like power spectrum density (PSD), common spatial pattern (CSP), and Riemannian geometry. 
We focused on the spatial distribution of the functional connectivity within and between hemispheres during motor imagery tasks, introducing network-based metrics such as integration and segregation. 
Evaluating these metrics across multiple EEG-based BCI datasets, our findings reveal that network lateralization offers neurophysiological plausible insights, characterized by stronger lateralization in sensorimotor and frontal areas contralateral to imagined movements. 
While these lateralization features did not outperform CSP and Riemannian geometry in terms of classification accuracy, they demonstrated competitive performance against PSD alone and provided biologically relevant interpretation. 
This study underscores the potential of brain network lateralization as a new feature to be integrated in motor imagery-based BCIs for enhancing the interpretability of noninvasive applications.


---
## Data
All data associated with this manuscript are publicly available and can be found in the [Mother of all BCI Benchmarks (MOABB)](http://moabb.neurotechx.com/docs/index.html) here:
[http://moabb.neurotechx.com/docs/datasets.html](http://moabb.neurotechx.com/docs/datasets.html)



## Code
This repository contains the code used to run the analysis performed and to plot the figures.
To install all the packages used in this work you can directy type in your terminal:
`pip install -r requirements.txt`



---
## Figures

### Figure 1 -   
