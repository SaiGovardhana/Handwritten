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


        if suggestions and len(suggestions) > 0:
            best_suggestion = suggestions[0]
            word=best_suggestion
    return word
