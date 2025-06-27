def get_extraction_prompt(prompt, response):
    return f"""
You are trying to verify how factual a piece of text is. To do so, you need to break down a language model's response and extract as many fine-grained facts mentioned in the response as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any **code**, **stories**, **personal experiences**, **hypotheticals** (e.g., "would be" or the subjunctive mood), **subjective statements** (e.g., opinions), **suggestions**, **advice**, **instructions**, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should describe either a single event (e.g., "Nvidia was founded in 1993 in Sunnyvale, California, U.S.") or a single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

You should focus on the named entities and numbers in the sentence and extract relevant information from it. Based on the context, you should recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and similar references. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name rather than by pronoun. Use the names of entities rather than definite noun phrases (e.g., "the teacher") whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., an embedded clause, a prepositional phrase, etc.). Each fact must be situated within a relevant temporal and locational context whenever needed. Keep each fact to one sentence, with zero or at most one embedded clause. You do not need to justify what you extract.

Copy the given **response** sentence by sentence in your output. After each sentence, extract the verifiable facts it contains and enclose them in <claim> and </claim> tags. Each fact should be written on a separate line. If a sentence does not contain any new verifiable atomic fact, copy the next sentence without adding an extra line break. Maintain the original line breaks from the response. If the whole response has no verifiable fact, simply copy everything in the given response without adding <claim> and </claim> tags.

The following are some examples. As the outputs are long, we use [...] to represent the omitted parts. Examples:

Example: Wikipedia
Response: The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae. Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable. The young shoots and leaves are sometimes eaten as greens.
Output:
The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae.
<claim>The scientific name of sweet potatoes is Ipomoea batatas.</claim>
<claim>Sweet potatoes are dicotyledonous plants.</claim>
<claim>Sweet potatoes belong to the bindweed or morning glory family, Convolvulaceae.</claim>
Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.
<claim>Sweet potatoes' roots are large.</claim>
<claim>Sweet potatoes' roots are starchy.</claim>
<claim>Sweet potatoes' roots are sweet-tasting.</claim>
<claim>Sweet potatoes' roots are tuberous.</claim>
<claim>Sweet potatoes' roots are used as a root vegetable.</claim>
The young shoots and leaves are sometimes eaten as greens. [...]

Example: Biography
Response: After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci. Both murals recorded military victories by the city (Michelangelo’s was the Battle of Cascina), but each also gave testimony to the special skills of the city’s much vaunted artists. Leonardo’s design shows galloping horses, Michelangelo’s active nudes—soldiers stop swimming and climb out of a river to answer an alarm.
Output:
After the success of the David in 1504, Michelangelo’s work consisted almost entirely of vast projects.
<claim>Michelangelo achieved the success of the David in 1504.</claim>
<claim>After 1504, Michelangelo’s work consisted almost entirely of vast projects.</claim>
He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished.
[...]
In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.
<claim>In 1504, Michelangelo agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall.</claim>
<claim>Around 1504, Leonardo da Vinci just began with a mural for the Florence city hall.</claim>
Both murals recorded military victories by the city (Michelangelo’s was the Battle of Cascina), but each also gave testimony to the special skills of the city’s much vaunted artists.
<claim>Michelangelo’s murals for the Florence city hall recorded military victories by the city.</claim>
<claim>Leonardo da Vinci’s murals for the Florence city hall recorded military victories by the city.</claim>
<claim>Michelangelo’s mural for the Florence city hall was the Battle of Cascina.</claim>
Leonardo’s design shows galloping horses, Michelangelo’s active nudes—soldiers stop swimming and climb out of a river to answer an alarm. [...]

Example: Experience
Response: I (27f) and my fiance "Leo" (27m) decided to let my FSIL "Maya" (32f) stay at our house because she needed space from her husband due to some relationship struggles they're having. Leo and I had gotten wedding cake samples from an expensive bakery specializing in wedding cakes. We planned to test them along with Maya after we finished up some other wedding plans yesterday. However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him. I asked what was happening, and Leo angrily told me that while we were both at work, Maya had some friends over and they ended up eating almost all of our cake samples.
Output:
[...] However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him. I asked what was happening, and Leo angrily told me that while we were both at work, Maya had some friends over and they ended up eating almost all of our cake samples. [...]

Example: Experience
Response: I was a catholic school kid, educated by nuns and somehow on a spring day in 1972, I was called down to the principal’s office by Sister Mary Roberts, who informed me that I had gained admission to Stuyvesant High School. I was excited to be freshman in one of New York City’s elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra. My parochial education in Science at St. Joseph’s was essentially “God made it, what else do you need to know?”
Output:
[...] I was excited to be freshman in one of New York City’s elite public schools but soon came to realize that my catholic school education did not provide the groundwork for abstract concepts like science and algebra.
<claim>Stuyvesant High School is in New York City.</claim>
<claim>Stuyvesant High School is an elite high school.</claim>
<claim>Stuyvesant High School is a public school.</claim>
<claim>In 1972, St. Joseph's catholic school education did not provide the groundwork for abstract concepts like science and algebra.</claim>
My parochial education in Science at St. Joseph’s was essentially “God made it, what else do you need to know?” [...]

Example: Medicine
Response: Major depressive disorder (MDD), also known as depression, is a mental disorder.
Output:
Major depressive disorder (MDD), also known as depression, is a mental disorder.
<claim>Major depressive disorder is also known as depression.</claim>
<claim>Major depressive disorder is a mental disorder.</claim>

Example: Wikipedia
Response: The 1937 Fox vault fire was a major fire in a 20th Century Fox film storage facility in Little Ferry, New Jersey on 9 July 1937. It was caused by the spontaneous combustion of nitrate film stored in inadequately-ventilated vaults. The fire resulted in one death and two injuries, and destroyed all of the film present. This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932. Also destroyed were Educational Pictures negatives and films of several other studios.
Output:
The 1937 Fox vault fire was a major fire in a 20th Century Fox film storage facility in Little Ferry, New Jersey on 9 July 1937.
<claim>Fox Film Corporation produced silent films before 1932.</claim>
<claim>The 1937 Fox vault fire caused the loss of most of the silent films produced by Fox Film Corporation before 1932.</claim>
It was caused by the spontaneous combustion of nitrate film stored in inadequately-ventilated vaults. [...]

Example: Biography
Response: Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there. When he said “you can’t get your youth back,” he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
Output:
Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.
<claim>Kevin Garnett spent over a decade with the Minnesota Timberwolves.</claim>
<claim>Kevin Garnett was loyal to the Minnesota Timberwolves.</claim>
<claim>Kevin Garnett found little success with the Minnesota Timberwolves.</claim>
When he said “you can’t get your youth back,” he meant it - because from a human standpoint, had he been able to apply his talents somewhere else, NBA history might have been different.
<claim>Kevin Garnett said "you can’t get your youth back."</claim>

Case: Quote
Response: Unity. Unity. In another January in Washington, on New Year’s Day 1863, Abraham Lincoln signed the Emancipation Proclamation. When he put pen to paper, the President said, “If my name ever goes down into history it will be for this act and my whole soul is in it.” My whole soul is in it.
Output:
[...] When he put pen to paper, the President said, “If my name ever goes down into history it will be for this act and my whole soul is in it.”
<claim>On New Year’s Day 1863, Abraham Lincoln said, “If my name ever goes down into history it will be for this act and my whole soul is in it.”</claim>

Case: Story
Response: Ãcariya Mun related the story of a dhutanga monk (ascetic monk) who inadvertently went to stay in a forest located next to a charnel ground. He arrived on foot at a certain village late one afternoon and, being unfamiliar with the area, asked the villagers where he could find a wooded area suitable for meditation. They pointed to a tract of forest, claiming it was suitable, but neglected to tell him that it was situated right on the edge of a charnel ground. They then guided him to the forest, where he passed the first night peacefully. On the following day he saw the villagers pass by carrying a corpse which they soon cremated only a short distance from where he was staying.
Output:
[...] They then guided him to the forest, where he passed the first night peacefully. On the following day he saw the villagers pass by carrying a corpse [...]

Case: Long Sentence
Response: Pope Julius had an ambitious imagination, parallel to Michelangelo’s, but because of other projects, such as the new building of St. Peter’s and his military campaigns, he evidently became disturbed soon by the cost. Michelangelo believed that Bramante, the equally prestigious architect at St. Peter’s, had influenced the pope to cut off his funds. He left Rome, but the pope brought pressure on the city authorities of Florence to send him back. He was put to work on a colossal bronze statue of the pope in his newly conquered city of Bologna (which the citizens pulled down soon after when they drove the papal army out) and then on the less expensive project of painting the ceiling of the Sistine Chapel (1508–12).
Output:
[...] He was put to work on a colossal bronze statue of the pope in his newly conquered city of Bologna (which the citizens pulled down soon after when they drove the papal army out) and then on the less expensive project of painting the ceiling of the Sistine Chapel (1508–12).
<claim>Michelangelo was put to work on a colossal bronze statue of Pope Julius II.</claim>
<claim>The city of Bologna was once conquered by Pope Julius II.</claim>
<claim>The colossal bronze statue of Pope Julius II was put in Bologna.</claim>
<claim>The papal army was driven out of Bologna.</claim>
<claim>The citizens of Bologna pulled down the bronze statue of Pope Julius II after they drove the papal army out.</claim>
<claim>Michelangelo worked on painting the ceiling of the Sistine Chapel from 1508 to 1512.</claim>

---

Extract *verifiable atomic* facts.

Prompt: {prompt}
Response: {response}
Output:
"""
