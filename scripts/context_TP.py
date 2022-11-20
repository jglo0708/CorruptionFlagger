from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
import numpy as np


text_file = "../outputs/test_eng_documents.txt"
nonengtext_file = "./outputs/test_noneng_documents.txt"

nltk.download('stopwords')

documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

logging.info(preprocessed_documents[:2])

tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

training_dataset = tp.create_training_set(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
logging.info(training_dataset.vocab[:10])

ctm = ZeroShotTM(input_size=len(tp.vocab), bert_input_size=512, n_components=50, num_epochs=15)
ctm.fit(training_dataset) # run the model

logging.info(ctm.get_topic_lists(5))
topics_predictions = ctm.get_thetas(training_dataset, n_samples=5) # get all the topic predictions


topic_number = np.argmax(topics_predictions[0]) # get the topic id of the first document

logging.info(ctm.get_topic_lists(5)[topic_number]) #and the topic should be about natural location related things

noneng_documents = [line.strip() for line in open(nonengtext_file, 'r').readlines()]

testing_dataset = tp.create_test_set(noneng_documents) # create dataset for the testset
noneng_topics_predictions = ctm.get_thetas(testing_dataset, n_samples=5) # get all the topic predictions
logging.info(italian_documents[0)

topic_number = np.argmax(noneng_topics_predictions[0]) # get the topic id of the first document
logging.info(ctm.get_topic_lists(10)[topic_number])