import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
cider = evaluate.load("cider")

def evaluate_captions(predictions, references):
    results = {
        "BLEU": bleu.compute(predictions=predictions, references=references),
        "METEOR": meteor.compute(predictions=predictions, references=references),
        "ROUGE-L": rouge.compute(predictions=predictions, references=references),
        "CIDEr": cider.compute(predictions=predictions, references=references),
    }
    return results
