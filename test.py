from model import Model
from predict import Predictor

M = Model()
predictor = Predictor()

text = """
I wouldn't name her burden
Or call him mistake
Or blame them on any
Pissed decision I've made

Respect for the man
Who pretended to be strong
And the woman who helped me
Find my song

Blood of a mutt
Whipped like a dog
Yes I'm a man
Though it's taken me so long
Formed by the fools
Who poisoned my guts
With a vengeance to prove it all wrong

Lying on the tracks of influence
Of every passing fling
Every judgement I make
Every song that I sing
Friends and foe
Every fork in the road
Add my tears to my tab
And leave me enough to get home

With my heart in my pocket
Both feet on the gas
Failure as my guide
Memories made of ash
Age by my side
So I'm never alone
Add my years to my tab
But leave me enough for the road
"""

prediction = predictor.predict([text])
