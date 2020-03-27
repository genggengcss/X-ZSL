
#### examples of rule pattern

Taking following classes as examples.
* Unseen class *Crab* (*dbr:Crab*) and its impressive seen class *Fiddler crab* (*dbr:Fiddler_crab*).
* Unseen class *Indian mongoose* (*dbr:Small_Asian_mongoose*) and its impressive seen class *Mongoose* (*dbr:Mongoose*).
* Unseen class *Guanaco* (*dbr:Guanaco*) and its impressive seen class *Llama* (*dbr:Llama*).
* Unseen class *Smooth hammerhead* (*dbr:Smooth_hammerhead*) and its impressive seen class *Hammerhead shark* (*dbr:Hammerhead_shark*).
* Unseen class *Wood ant* (*dbr:Formica*) and its impressive seen class *Ant* (*dbr:Ant*).
* Unseen class *Guanaco* (*dbr:Guanaco*) and its impressive seen class *Llama* (*dbr:Llama*).

|Triple Pattern|Diagram|Examples|Illustration|
|----|-----|----|-----|
|$ (s, r, u)$|<img src="img/rule1.png" width = "80"/>|(*dbr:Fiddler_crab, dbo:order, dbr:Crab*)|*Fiddler crab* is directly related with *Crab* via relation *hypernym*.|
|$(u, r, s)$|<img src="img/rule2.png" width = "80"/>|(*dbr:Small_Asian_mongoose, hypernym, dbr:Mongoose*)|*Indian mongoose* is directly related with *Mongoose* via relation *hypernym*.|
|<img src="img/code1.png" width = "100"/>|<img src="img/rule3.png" width = "80"/>|(*dbr:Guanaco, hypernym, dbr:Camelid*) & <br> (*dbr:Llama, hypernym, dbr:Camelid*)<br><br>(*dbr:Smooth_hammerhead, hypernym, dbr:Shark*) & <br> (*dbr:Hammerhead_shark, dbo:order, dbr:Shark*)|*Guanaco* and *Llama* are both the members of *Camelid*. <br><br><br> *Smooth hammerhead* and *Hammerhead shark* are both relevant to *Shark* via relation *hypernym* and *order*.|
|$(s, p, v) \wedge (u, p, v)$|<img src="img/rule4.png" width = "80"/>|(*dbr:Formica, dbp:typeSpecies, Formica rufa*) &<br> (*dbr:Ant, dbp:typeSpecies, Formica rufa*)|*Wood ant* and *Ant* both have property *species type* and share the same property value *Formica rufa*.|
|$$ (s, r_1, t) \wedge (t, r_2, u)$$|<img src="img/rule5.png" width = "80"/>|(*dbr:Fiddler_crab, dbo:family, dbr:Ocypodidae*) &<br>(*dbr:Ocypodidae, dbo:order, dbr:Crab*)|*Fiddler crab* and *Crab* is related via a transitional entity *Ocypodidae*.|

