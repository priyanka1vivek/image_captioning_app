# Image Captioning App

This combines analysis with image captioning using the Salesforce BLIP model and Hugging Face's Transformers library. It provides captions for uploaded images on the basis of categories like Greedy, Beam, or Nucleus algorithms.
Evaluation Dataset: MS COCO (Microsoft Common Objects in Context) https://cocodataset.org

*Model Selection:*
A single, end-to-end BLIP (Bootstrapping Language-Image Pre-training) Transformer model (Salesforce/blip-image-captioning-large).
BLIP is a unified vision-language model that processes images and text jointly. This allows it to learn far richer connections between visual concepts and words, resulting in more accurate and context-aware captions. 

*Evaluation Metrics:* 
To properly measure the quality of our model, we will expand the evaluation beyond a single, limited metric.
o	BLEU (1-4): For baseline precision comparison.

o	METEOR: For semantic accuracy, as it recognizes synonyms and stemmed words.

o	ROUGE-L: For recall, ensuring the main ideas are captured.

o	CIDEr: The primary metric for this task, designed to measure how well a caption matches human consensus.
