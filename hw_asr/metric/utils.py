# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    if len(target_text) == 0 and len(predicted_text) == 0:
        return 1
    editdist = editdistance.eval(target_text, predicted_text)
    return editdist / len(target_text)



def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    # raise NotImplementedError()
    if len(target_text.split()) == 0 and len(predicted_text.split()) == 0:
        return 1
    editdist = editdistance.eval(target_text.split(), predicted_text.split())
    return editdist / len(target_text.split())