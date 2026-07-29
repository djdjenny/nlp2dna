
## Изначальная идея

- BPE вычислительно дешево, но размывает истинный сигнал
- Выявить, какая буква в BPE наиболее важная
- Берем in silico mutagenesis и делаем с его помощью корректор предсказания для токенов с высокой важностью (предположительно Integrated Gradients или Attention Rollout - пока не шарю за них)

[https://www.biorxiv.org/content/10.1101/2023.11.10.566588v1](https://www.biorxiv.org/content/10.1101/2023.11.10.566588v1) (2024) - ISM, но еще эффективнее, не в 3 прохода, а в 1

[https://link.springer.com/article/10.1007/s00122-025-04973-1](https://link.springer.com/article/10.1007/s00122-025-04973-1) (2025) - ревью существующих методов направленного мутагенез для выведения растений с определенным фенотипом, есть белковые и венозные, но с фокусом на растениях - из перспективных направлений выделяется улучшение предсказания на регуляторных декодирующих последовательностях

https://www.biorxiv.org/content/10.1101/2025.04.16.648420v1 - Identifying non-coding variant effects at scale via machine learning models of cis-regulatory reporter assays - решает проблему некодирующих вариантов на человеке, валидация на тех же данных, которые хотела смотреть я

https://www.biorxiv.org/content/10.1101/2025.12.01.691503v1.full - MutBPE - похоже, но микроразница в том, что там выбираются координаты мутация (?)

>Future work will explore the generalizability of Mut-BPE across a broader range of BPE-based genomic models. In addition, we aim to investigate the underlying mechanisms by which Mut-BPE improves variant representations, through embedding visualization, attention pattern analysis, and gradient-based interpretability techniques.


DNAChunker - разница в том, что это надстройка над моделью, а не новая архитектура


https://arxiv.org/abs/2606.06834 - регулом, представлен новый метод - residualization-and-permutation diagnostic. Отделяем эффект вариации регуляторных элементов от predictability-driven variance - языковые модели (кадеус, гиена) ставят более высокий скор транспозонам - длинным +- константным элементам, функциональные модели (Enformer) лучше для выявления регуляторных сигналов.


----

## Идея #2 
Повторить https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013424
Paying attention to attention: High attention sites as indicators of protein family and function in language models 
на геномике

Нюансы - меньше информации в отдельном нуклеотиде, "опорные точки" более расплывчаты, контекст важнее, чем в белках.


-----

## Атаки на геномные модели 

https://arxiv.org/abs/2603.27465 - training data poisoning in genomic language models, targeting both pre-training and fine-tuning stages. 
At pre-training, using Evo 2 and GENERator architectures, we show that less than 1% adversarially crafted sequences in the training corpus can selectively degrade generative performance on targeted genomic contexts while leaving unrelated sequences unaffected. We evaluate three scenarios: corruption of TATA-box promoter motifs, disruption of CTCF binding sites, and insertion of synthetic sequences absent from all training genomes. At fine-tuning, we demonstrate two additional attacks. First, poisoning a subset of CTCF sites in a ClinVar-derived corpus installs a conditional backdoor in a LoRA-adapted model that activates almost exclusively when the trigger sequence is present. Second, using frozen Evo 2 7B embeddings, targeted label corruption of downstream training data selectively compromises a clinically relevant variant classification task, demonstrated on BRCA1 variant effect prediction. These results show genomic foundation models are susceptible to targeted data poisoning with minimal footprint.

https://arxiv.org/abs/2506.00821 - FGSM (Fast Gradient Sign Method) для внесения минимальных пертурбаций и создания "зараженных" копий генов -> значительно ухудшаем качество модели. Также рассматривали soft prompt атаки на эмбеддинги, тоже все портят.

---

## Мутации

Статьи:

https://www.nature.com/articles/s41467-025-65823-8 - бенчмарк, в котором упоминается патогенность вариантов как отдельная задача. Как смотрели - разница эмбеддингов референса и мутации как фича для случайного леса.

https://arxiv.org/abs/2604.04287 - Entropy, Disagreement, and the Limits of Foundation Models in Genomics

> Foundation models in genomics have shown mixed success compared to their counterparts in natural language processing. Yet, the reasons for their limited effectiveness remain poorly understood. In this work, we investigate the role of entropy as a fundamental factor limiting the capacities of such models to learn from their training data and develop foundational capabilities. We train ensembles of models on text and DNA sequences and analyze their predictions, static embeddings, and empirical Fisher information flow. We show that the high entropy of genomic sequences -- from the point of view of unseen token prediction -- leads to near-uniform output distributions, disagreement across models, and unstable static embeddings, even for models that are matched in architecture, training and data. We then demonstrate that models trained on DNA concentrate Fisher information in embedding layers, seemingly failing to exploit inter-token relationships. Our results suggest that self-supervised training from sequences alone may not be applicable to genomic data, calling into question the assumptions underlying current methodologies for training genomic foundation models.

>Using entropy as a starting point, we performed a simple experiment consisting in comparing BERT models trained on English text to identical models trained on DNA sequences. We proposed two main directions that we believe deserve deeper exploration in future work. First, high data entropy leads to uncertain predictions, but also to disagreement among models, even under matched training methodology, data and architecture. Second, analysis of aggregated empirical Fisher information suggests that DNA models seemingly fail to effectively capture inter-token relationships, as information is concentrated in static embeddings rather than in transformer layers. We also find that using different tokenization schemes for DNA had little impact in all of the reported metrics, further suggesting that our results reflect fundamental properties of the data itself.


https://www.biorxiv.org/content/10.1101/2024.12.18.628606v3.full - обзор необходимости претрейна на геномных моделях, среди прочего
>We also find that the evaluated GFMs fail to capture clinically relevant genetic mutations, with embeddings and log-likelihood ratios showing limited sensitivity to annotated variants.

> We identify areas for methodological refinement, including optimizing masking approach, employing character-level tokenization, and designing specialized architectures better attuned to biological sequence complexity.


https://arxiv.org/abs/2507.05265 - SNP-ориентированная модель ДНК

>For each genomic position that has variants, we averaged available population-specific variation frequencies and built a genome-wide variation frequency matrix. For each genomic position, this matrix provides the multinomial distribution of 11 probabilities on 5 types of nucleotides (A, C, G, T, N), 5 types of insertions (insertions after A, C, G, T, N), and deletion.
>similar to the reference genome, we created random DNA sequence samples from the variation-encoded genome. Specifically, for each genomic position with variations, we first sampled a biallelic representation of two possible nucleotides, insertions, or deletions without replacement from the multinomial distribution of the variation frequency matrix (see Figure 1). Then we mapped each variant or nucleotide-pair of variants to a unique single Chinese character for a total of 121 characters, all extracted from the classic poem Li Sao.


https://www.biorxiv.org/content/10.64898/2026.05.31.729117v1 - Mechanistic Annotation of Genomic Impacts - уходим от черного ящика - объясняем патогенные варианты через логический слой, который интегрирует очень много данных. В результате может работать с новыми патологическими вариантами и давать биологические гипотезы, почему они патогенные.

----

## Идея 3. Геометрическая интерпретация мутаций

Гипотеза - изменение эмбеддинга при мутации нужно рассматривать не как "эта мутация плохая", а как геометрию - разница должна коррелировать с величиной эффекта (например, изменение экспрессии), схожие мутации должны давать схожие изменения, по нескольким мутациям можно измерять эпистаз.

Шаги
- Baseline и данные: взять [Sharon et al. 2012](https://www.nature.com/articles/nbt.2205) (промоторный MPRA с эпистазом) и yeast segregant eQTL панели; получить эмбеддинги/SAE фичи (тут опять же пока не очень разбираюсь)
- Посчитать расстояние WT->mutant, сравнить с эффектом
- Сгруппировать сдвиги по типу мутации (нарушение конкретного мотива), проверить, кластеризуются ли направления/SAE фичи
- Для двойных мутантов сравнить сумму двух одиночных сдвигов с реальным сдвигом двойного мутанта
- Контроль - сравнить всё с likelihood-ratio и простым motif-disruption score — убедиться, что сигнал не сводится к более простому методу.
- Дополнительно - провалидироваться на арабидопсисе, чтобы понять, насколько генерализуемо. 

Потенциальные проблемы
1. Какой конкретно эмбеддинг брать? В Evo2 SAE брали 26й слой - можем ли мы сделать то же самое?
2. Если берем SAE, то лочимся в модель и чекпоинт??
3. Можем не превзойти log-likelihood по предсказанию
4. Потенциально мало эпистатических клонов
5. Это только корреляционный анализ

Что тут уже сделано

GPN
Считает log-likelihood ratio и считает патогенность - выдает 1 число на вариант, но геометрию не смотрит.

GPN-MSA
Использует выравнивание (эволюционный консерватизм) - ттоже получает число.

[Shorkie](https://www.biorxiv.org/content/10.1101/2025.09.19.677475)
Связывает последовательность и экспрессию, использует MPRA. Это finetuned под экспрессию моделька, а не исследование эмбеддингов изначальной модели, но отсюда можно взять датасеты.

Evo2
Универсальная модель, показано выучивание биологических признаков (например, связывания транскрипционных факторов). Исследуем геометрию сдвига, в тч корреляцию с величиной эффекта.

SAE Evo2
Нашли отдельные интерпретируемые признаки - качественно. Мой вопрос - коррелирует ли активация признака с величиной эффекта?


TBD - надо нормально ознакомиться, если решим заниматься
https://arxiv.org/abs/2602.17532
?? пример того, как можно строить процесс с валидацией, что мы превосходим бейзлайн?

https://www.biorxiv.org/content/10.1101/2025.09.14.676130v1
исследование для белков, в котором нашли эпистаз 

https://www.biorxiv.org/content/10.64898/2026.04.23.719915v1 
критика белковой модели с эпистазом - MULTI-evolve, хорошее описание бейзлайна

https://arxiv.org/abs/2504.10388
надо внимательно посмотреть - тоже исследуют эпистаз в геномике, но учат свою маленькую модель
