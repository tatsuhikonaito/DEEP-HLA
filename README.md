<img width="800" alt="DEEP-HLA_logo" src="https://user-images.githubusercontent.com/67576643/86470256-08398780-bd76-11ea-9b53-be7c05e2add7.png">

# DEEP*HLA

`DEEP*HLA`  is an HLA allelic imputation program based on a multi-task convolutional neural network implemented in Python.

## Overview

`DEEP*HLA` receives SNV data and outputs genotype dosages of binary HLA alleles. In  `DEEP*HLA`, HLA imputation is peformed in two processes: (1) a model training with a HLA reference panel and (2) Imputation with a trained model.

## Requirements

- Python 3.x (3.7.4)
- Pytorch (1.4.0)
- Numpy (1.17.2)
- Pandas (0.25.1)
- Scipy (1.3.1)
- Argparse (1.4.0)

`DEEP*HLA` was tested on the versions in parentheses, so we do not guarantee that it will work on different versions.

## Installation

Just clone this repository as folllows.

```
git clone https://github.com/tatsuhikonaito/DEEP-HLA
cd ./DEEP-HLA
```



## Usage

### 0. Original file formats

 The original files for model and HLA information are needed to run `DEEP*HLA`.

* {MODEL}.config.json

  The description of a model configuration, including grouping of HLA genes, window size of SNV (Kb), and parameters of neural networks. The gene names must be consistent with reference data.

  ```
  {
  	"group1": {
  		"HLA": ["HLA_F", "HLA_G", ...],
  		"w": 500,
  		"conv1_num_filter": 128,
  		"conv2_num_filter": 64, 
  		"conv1_kernel_size": 64, 
  		"conv2_kernel_size": 64, 
  		"fc_len": 256
  	},
  	"group2": {
  		"HLA": ["HLA_C", "HLA_B", ...],
  		"w": 500,
  		"conv1_num_filter": 128,
  		"conv2_num_filter": 64, 
  		"conv1_kernel_size": 64, 
  		"conv2_kernel_size": 64, 
  		"fc_len": 256
  	},	
  	...
  }
  ```


* {HLA}.info.json

  The description of information of HLA genes in reference data, including HLA gene names, position, and HLA allele names for each resolution. They must be consistent with reference data.

  ```
  {
  	"HLA_F": {
  		"pos": 29698429,
  		"2-digit": ["HLA_F_01", ...],
  		"4-digit": ["HLA_F_01:01", "HLA_F_01:03", ...]
  	},
  	"HLA_G": {
  		"pos": 29796823,
  		"2-digit": ["HLA_G_01", ...],
  		"4-digit": ["HLA_G_01:01", "HLA_G_01:03", ...]
  	},
  	...
  }
  ```




### 1. Model training

Run  `train.py` on a command-line interface as follows. 

```
$ python train.py --ref REFERENCE (.bgl.phased/.bim) --sample SAMPLE (.bim) --model MODEL (.config.json) --hla HLA (.info.json) --model-dir MODEL_DIR
```

##### Arguments and options

| Option name   | Descriptions                                                 | Required | Default   |
| ------------- | ------------------------------------------------------------ | -------- | --------- |
| `--ref`       | HLA reference data (.bgl.phased, and .bim format).           | Yes      | None      |
| `--sample`    | Sample SNP data (.bim format).                               | Yes      | None      |
| `--model`     | Model configuration (.config.json format).                   | Yes      | None      |
| `--hla`       | HLA information of the reference data (.info.json format).   | Yes      | None      |
| `--model-dir` | Directory for saving trained models.                         | No       | "model"   |
| `--num-epoch` | Number of epochs to train.                                   | No       | 100       |
| `--patience`  | Patience for early-stopping. If you prefer no early-stopping, specify the same value as `--num-epoch`. | No       | 16        |
| `--val-split` | Ratio of splitting data for validation.                      | No       | 0.1       |
| `--max-digit` | Maximum resolution of alleles to impute ("2-digit", "4-digit", or "6-digit"). | No       | "4-digit" |

##### Outputs

- {MODEL_DIR}/{group}\_{digit}\_{hla}.pickle

  Trained models. 

- {MODEL_DIR}/model.bim

  SNP infomation used in tranining and subsequent imputation process.

- {MODEL_DIR}]/best_val.txt

  Accuracies of trained models in validation process. 


### 2. Imputation

After you have finished training a model, run `impute.py` as follows. 

```
$ python impute.py --sample SAMPLE (.bgl.phased (.haps)/.bim) --model MODEL (.config) --hla HLA (.info.json) --model-dir MODEL_DIR --out OUT
```

##### Arguments and options

| Option name     | Descriptions                                                 | Required | Default   |
| --------------- | ------------------------------------------------------------ | -------- | --------- |
| `--sample`      | Sample SNP data (.bgl.phased or .haps, .bim, and .fam format). | Yes      | None      |
| `--phased-type` | File format of sample phased file ("bgl" or "hap").            | No       | "bgl"     |
| `--model`       | Model configuration (.config.json and .bim format).          | Yes      | None      |
| `--hla`         | HLA information of the reference data (.info.json format).   | Yes      | None      |
| `--model-dir`   | Directory where trained models are saved.                    | No       | "model"   |
| `--out`         | Prefix of output files.                                      | Yes      | None      |
| `--max-digit`   | Maximum resolution of alleles to impute ("2-digit", "4-digit", or "6-digit"). | No       | "4-digit" |
| `--mc-dropout`  | Whether to calculate uncertainty by Monte Carlo dropout (True or False). | No       | False     |

##### Outputs

- {OUT}.deephla.phased

  Imputed allele phased (best-guess genotypes) data. 

  Rows are markers and columns are individuals.

  First column is marker name; and subsequent columns are genotypes as two columns per individual.

- {OUT}.deephla.dosage

  Imputed allele dosage data. 

  First, second, and third columns are marker name, ref ("P"), and alt ("A"); and subsequent columns are dosages as one column per individual.

  Rows are markers and columns are individuals, as one column per individual. 

- {OUT}.deephla.entropy  (optional)

  Uncertainty based on entropy of sampling variation in Monte Carlo dropout.

  First column is marker name; and subsequent columns are entropys as one column per individual.

## Sample

The sample models used in our study will be uploaded after the orignal paper of `DEEP*HLA` is published.

## Reference

`DEEP*HLA` uses MGDA-UB (Multiple Gradient Descent Algorithm - Upper Bound) for multi-task learning, and the source code of its part refers to [MultiObjectiveOptimization](https://github.com/intel-isl/MultiObjectiveOptimization). 

## Contact

For any question, you can contact Tatsuhiko Naito ([tnaito@sg.med.osaka-u.ac.jp](mailto:tnaito@sg.med.osaka-u.ac.jp))
