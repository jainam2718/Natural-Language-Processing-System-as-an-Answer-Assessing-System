import pandas as pd
from dataAugmentation.backtranslation import backTranslation
from dataAugmentation.synonymReplacement import augment_samples

biosses_df = pd.read_csv("data/biosses_dataset.csv")

# biosses_df['sentence1'] = backTranslation(biosses_df['sentence1'])
# biosses_df['sentence2'] = backTranslation(biosses_df['sentence2'])

# biosses_df.to_csv("./data/biosses_back_translation_augmented.csv")

# df = pd.read_csv("data/biosses_back_translation_augmented.csv")

biosses_df['sentence1'] = augment_samples(biosses_df['sentence1'])
biosses_df['sentence2'] = augment_samples(biosses_df['sentence2'])

biosses_df.to_csv("./data/biosses_synonym_replacement_augmented.csv")