---
license: mit
language:
- en
metrics:
- accuracy
- f1
- precision
- recall
library_name: transformers
---
# MatSciBERT
## A Materials Domain Language Model for Text Mining and Information Extraction

This is the pretrained model presented in [MatSciBERT: A materials domain language model for text mining and information extraction](https://rdcu.be/cMAp5), which is a BERT model trained on material science research papers.

The training corpus comprises papers related to the broad category of materials: alloys, glasses, metallic glasses, cement and concrete. We have utilised the abstracts and full text of papers(when available). All the research papers have been downloaded from [ScienceDirect](https://www.sciencedirect.com/) using the [Elsevier API](https://dev.elsevier.com/). The detailed methodology is given in the paper.

The codes for pretraining and finetuning on downstream tasks are shared on [GitHub](https://github.com/m3rg-repo/MatSciBERT).

If you find this useful in your research, please consider citing:
```
@article{gupta_matscibert_2022,
  title   = "{MatSciBERT}: A Materials Domain Language Model for Text Mining and Information Extraction",
  author  = "Gupta, Tanishq and 
            Zaki, Mohd and 
            Krishnan, N. M. Anoop and 
            Mausam",
  year    = "2022",
  month   = may,
  journal = "npj Computational Materials",
  volume  = "8",
  number  = "1",
  pages   = "102",
  issn    = "2057-3960",
  url     = "https://www.nature.com/articles/s41524-022-00784-w",
  doi     = "10.1038/s41524-022-00784-w"
}
```