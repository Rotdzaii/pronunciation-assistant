\# CMU ARCTIC Dataset



This folder is used for preparing CMU ARCTIC as the native English reference dataset.



\## Purpose



CMU ARCTIC is used as native/reference English speech data for pronunciation training experiments.



In this project, CMU ARCTIC will be combined with L2-ARCTIC Vietnamese speakers to build a baseline classifier:



\- `0 = native\_reference`

\- `1 = non\_native\_learner`



\## Local folder structure



```txt

datasets/cmu-arctic/

&#x20; raw/          # original downloaded dataset, ignored by Git

&#x20; processed/    # normalized wav files, ignored by Git

&#x20; metadata/     # generated lightweight metadata files

