
def get_veriscore_extractor_prompt_ft(snippet):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

### Input:
{snippet}

### Response:""".strip()


def get_veriscore_extractor_prompt_gpt(snippet, sentence):
    return f"""
You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract as many fine-grained facts mentioned in the sentence as possible. Each of these fine-grained facts should be verifiable against reliable external world knowledge (e.g., via Wikipedia). Any story, personal experiences, hypotheticals (e.g., "would be" or subjunctive), subjective statements (e.g., opinions), suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should also be describing either one single event (e.g., "Nvidia is founded in 1993 in Sunnyvale, California, U.S.") or single state (e.g., "UMass Amherst has existed for 161 years.") with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. Other sentences are only context for you to recover pronouns, definite phrases (e.g., "the victims" or "the pope"), and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name but not pronoun. Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). Each fact must be situated within relevant temporal and location whenever needed. Keep each fact to one sentence with zero or at most one embedded clause. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, please write "No verifiable claim."

Extract *verifiable atomic* facts. Write one fact per line. Do not include any numbering or bullet points.

Text: {snippet}
Sentence to be focused on: {sentence}
Facts:""".strip()