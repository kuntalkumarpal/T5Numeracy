# T5Numeracy

Testing T5 models for Numeracy Skills through four Tasks : 
* Numeration, 
* Magnitude Order Prediction, 
* List MinMax  
* List Sorting (Ascending, Descending)

This repository contains the code for the following paper accepted in "_EMNLP2021-Findings_" and in ["_The Second Workshop on Insights from Negative Results in NLP_"](https://insights-workshop.github.io/)

# Paper

[Investigating Numeracy Learning Ability of a Text-to-Text Transfer Model](https://aclanthology.org/2021.findings-emnlp.265)

## Abstract
The transformer-based pre-trained language models have been tremendously successful in most of the conventional NLP tasks. But they often struggle in those tasks where numerical understanding is required. Some possible reasons can be the tokenizers and pre-training objectives which are not specifically designed to learn and preserve numeracy. Here we investigate the ability of text-to-text transfer learning model (T5), which has outperformed its predecessors in the conventional NLP tasks, to learn numeracy. We consider four numeracy tasks: numeration, magnitude order prediction, finding minimum and maximum in a series, and sorting. We find that, although T5 models perform reasonably well in the interpolation setting, they struggle considerably in the extrapolation setting across all four tasks.

## Reference

```
@inproceedings{pal-baral-2021-investigating-numeracy,
    title = "Investigating Numeracy Learning Ability of a Text-to-Text Transfer Model",
    author = "Pal, Kuntal Kumar  and
      Baral, Chitta",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.265",
    pages = "3095--3101",
    abstract = "The transformer-based pre-trained language models have been tremendously successful in most of the conventional NLP tasks. But they often struggle in those tasks where numerical understanding is required. Some possible reasons can be the tokenizers and pre-training objectives which are not specifically designed to learn and preserve numeracy. Here we investigate the ability of text-to-text transfer learning model (T5), which has outperformed its predecessors in the conventional NLP tasks, to learn numeracy. We consider four numeracy tasks: numeration, magnitude order prediction, finding minimum and maximum in a series, and sorting. We find that, although T5 models perform reasonably well in the interpolation setting, they struggle considerably in the extrapolation setting across all four tasks.",
}
```
