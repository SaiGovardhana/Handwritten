import enchant
def postProcess(word):
    if word == None or word=="":
        return ""
    d = enchant.Dict("en_US")
    if d.check(word):
        pass
    else:
        suggestions = d.suggest(word)
        suggestions = [x for x in suggestions if not any(c.isspace() for c in x)]


        if suggestions:
            best_suggestion = suggestions[1]
            word=best_suggestion
    return word
