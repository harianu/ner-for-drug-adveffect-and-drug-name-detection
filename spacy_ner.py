from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import config


# New entity labels

LABEL = ['O','B-DRUG','B-ADVEFFECT','I-ADVEFFECT','I-DRUG']


# Loading training data 
with open ('ner_custom', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='new_model', output_dir=config.input_file_dict['nermodel'], n_iter=15):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,losses=losses)
            print('Losses', losses)

    # Save model 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)



if __name__ == '__main__':
    plac.call(main)


 # Test the trained model
    test_text = 'We report two new cases of sarcoidosis in two patients with hepatitis C virus infection treated with interferon alfa and ribavirin.'
    nlp = spacy.load(config.input_file_dict['nermodel'])
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)


 # Test the saved model

test_text2 = '''CONCLUSIONS: Itraconazole-induced liver injury presents with a cholestatic pattern of injury with damage to the interlobular bile ducts, possibly leading to ductopenia.
'''

nlp2 = spacy.load(config.input_file_dict['nermodel'])

print("Loading from", config.input_file_dict['nermodel'])
doc2 = nlp2(test_text2)
print(doc2.ents)
print(len(doc2.ents))
for ent in doc2.ents:
        print(ent.label_, ent.text,ent.start_char,ent.end_char)








