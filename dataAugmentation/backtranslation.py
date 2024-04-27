
from transformers import MarianMTModel, MarianTokenizer

def format_batch_texts(language_code, batch_texts):
  
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

    return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    return translated_texts

def english_to_french(sentences):

    # Get the name of the first model
    enFrModelName = 'Helsinki-NLP/opus-mt-en-fr'

    # Get the tokenizer
    enFrModelTokenizer = MarianTokenizer.from_pretrained(enFrModelName)

    # Load the pretrained model based on the name
    enFrModel = MarianMTModel.from_pretrained(enFrModelName)

    translated_texts = perform_translation(sentences, enFrModel, enFrModelTokenizer)

    return translated_texts

def french_to_english(sentences):

    # Get the name of the second model
    frEnModelName = 'Helsinki-NLP/opus-mt-fr-en'

    # Get the tokenizer
    frEnTokenizer = MarianTokenizer.from_pretrained(frEnModelName)

    # Load the pretrained model based on the name
    frEnModel = MarianMTModel.from_pretrained(frEnModelName)

    translated_texts = perform_translation(sentences, frEnModel, frEnTokenizer)

    return translated_texts

def backTranslation(sentences):

    translated_texts = english_to_french(sentences)
    back_tranlated_texts = french_to_english(translated_texts)
    return back_tranlated_texts