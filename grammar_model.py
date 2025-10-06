from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class GrammarCorrector:
    def __init__(self, model_name="prithivida/grammar_error_correcter_v1"):
        print("Loading model... please wait")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def correct(self, text):
        input_text = "gec: " + text
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
