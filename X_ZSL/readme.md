
#### examples of rule pattern

Taking unseen class *Crab* and its impressive seen classes *Fiddler_crab* as examples.
unseen class *Indian mongoose* and its impressive seen classes *Mongoose*
|Triple Pattern|Diagram|Examples|Illustration|
|----|-----|----|-----|
|$ (s, r, u)$|<img src="img/rule1.png" width = "80"/>|(*dbr:Fiddler_crab, dbo:order, dbr:Crab*)|*Fiddler crab* is directly related with *Crab* via the relation *hypernym*.|
|$(u, r, s)$|<img src="img/rule2.png" width = "80"/>|(*dbr:Small_Asian_mongoose, hypernym, dbr:Mongoose*)|*Indian mongoose* is directly related with *Mongoose* via the relation *hypernym*.|
|<img src="img/code1.png" width = "100"/>|<img src="img/rule3.png" width = "80"/>|
|$(s, p, v) \wedge (u, p, v)$|<img src="img/rule4.png" width = "80"/>|
|$$ (s, r_1, t) \wedge (t, r_2, u)$$|<img src="img/rule5.png" width = "80"/>|(*(dbr:Fiddler_crab, dbo:family, dbr:Ocypodidae)*   *(dbr:Ocypodidae, dbo:order, dbr:Crab)*)||

