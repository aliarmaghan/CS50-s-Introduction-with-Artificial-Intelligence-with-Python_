from logic import *

rain = Symbol("rain")   # It is raining.
hagrid = Symbol("hagrid")   # Harry visited Hagrid.
dumbledore = Symbol("dumbledore")   # Harry visited Dumbledore.

sentence = And(rain,hagrid)     # test


knowledge = And(
    Implication(Not(rain), hagrid),
    Or(hagrid, dumbledore),
    Not(And(hagrid, dumbledore)),
    dumbledore
)

# print(sentence.formula())
print(knowledge.formula())
print(model_check(knowledge, rain))
