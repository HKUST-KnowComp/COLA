"""
utils.py

"""
import numpy as np

"""
generate temporal probabilities
"""
import string
def crop_sent(s, spacy_model, sent_idx=1, offset=3):
    s = list(spacy_model(s).sents)[sent_idx][offset:].text
    s = s.strip(string.punctuation).replace('\n', ' ').strip() + '.'
    return s[0].upper() + s[1:]
    

def glt_get_probs(s, model, spacy_model, top_k=5):
    D, Y, X, Dint = s[['text', 'outcome', 'covariates', 'interventions']]
    if "covariates_cleaned" in s:
        print(f"{s['index']}: using cleaned X")
        X = s['covariates_cleaned']
    else:
        X = [crop_sent(x, spacy_model=spacy_model) for x in X]

    # p(x, d)
    baseln_probs = [model.get_temp(x,"",top_k=top_k) for x in X]
    tmp_probs=[[model.get_temp(x,d,top_k=top_k)+baseln_probs[xidx] for d in [D]+Dint]
               for xidx, x in enumerate(X)]

    # p(d, y)
    tmp_y_probs = [model.get_temp(d,Y,top_k=top_k) for d in [D]+Dint]

    # p(x, y)
    tmp_xy_probs = [model.get_temp(x,Y,top_k=top_k) for x in X]

    s['p_xd'] = tmp_probs
    s['p_dy'] = tmp_y_probs
    s['p_xy'] = tmp_xy_probs
    return s


def copa_get_probs(s, model, spacy_model, top_k=5):
    D, X, Dint, Y = s['text'], s['covariates'], s['interventions'], s['outcome']
    new_X = []
    for x in X:
        new_X.append(crop_sent(x, spacy_model=spacy_model))
    X = new_X

    # p(x, d)
    baseln_probs = [model.get_temp(x,"",top_k=top_k) for x in X]
    tmp_probs=[[model.get_temp(x,d,top_k=top_k)+baseln_probs[xidx] for d in [D]+Dint]
               for xidx, x in enumerate(X)]

    # p(d, y)
    tmp_y_probs = [model.get_temp(d,Y,top_k=top_k) for d in [D]+Dint]

    # p(x, y)
    tmp_xy_probs = [model.get_temp(x,Y,top_k=top_k) for x in X]

    s['p_xd'] = tmp_probs
    s['p_dy'] = tmp_y_probs
    s['p_xy'] = tmp_xy_probs
    return s

"""
testing temporal predictor on COPA
"""
def test_copa_run(ds, base_pred, ft_pred, top_k=5):
    def _res_proc(res):
        return f"before: {res[0]:.3f}\tafter: {res[1]:.3f}"
    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2', 'question', 'label']]
    print(f"Premise: {premise}\nC1: {choice1}\nC2: {choice2}\nQuestion: {q}\tCorrect choice: Choice {lb+1}")
    base_res = [base_pred.get_temp(premise, choice1, top_k=top_k), base_pred.get_temp(premise, choice2, top_k=top_k)]
    ft_res = [ft_pred.get_temp(premise, choice1, top_k=top_k), ft_pred.get_temp(premise, choice2, top_k=top_k)]
    exp = " (expect before > after)" if q == 'effect' else " (expect before < after)"
    print(f"\n============== PREMISE <---> CHOICE 1{exp if lb == 0 else ''}\n{premise} <---> {choice1}")
    print(f"Base model:\t{_res_proc(base_res[0])}\nFT model:\t{_res_proc(ft_res[0])}")
    print(f"\n============== PREMISE <---> CHOICE 2{exp if lb == 1 else ''}\n{premise} <---> {choice2}")
    print(f"Base model:\t{_res_proc(base_res[1])}\nFT model:\t{_res_proc(ft_res[1])}")

def test_copa_predict(ds, predictor, top_k=5):
    def relu(x): return np.maximum(0., x)
    premise, choice1, choice2, q, lb = ds[['premise', 'choice1', 'choice2', 'question', 'label']]
    res = [predictor.get_temp(premise, choice1, top_k=top_k), predictor.get_temp(premise, choice2, top_k=top_k)]
    befores = [r[0] - r[1] for r in res]
    afters = [r[1] - r[0] for r in res]


    if q == 'effect':
        # wants to find one with higher "AFTER"
        if afters[0] == afters[1]:
            print(f"tie at {premise}/after: {afters[0]}")
            return -1
        return np.argmax(afters)
    else:
        # asks for "cause", wants to find one with higher "BEFORE"
        if befores[0] == befores[1]:
            print(f"tie at {premise}/before: {befores[0]}")
            return -1
        return np.argmax(befores)
