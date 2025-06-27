def get_verification_prompt_no_context(claim):
    return f"""
Please generate only the letter A or B as the answer.  
Question: Is the following claim correct or incorrect?  
A. True  
B. False  

Claim: {claim}  
Answer:
""".strip()


def get_verification_prompt_fava(passage, query, doc):
    return f"""
Given a passage with factual errors, identify any entity, relation, contradictory errors in the passage. If there are no errors, return the passage with no tags. Any changes to the original passage should be marked in <> tags. Below are the error definitions followed by examples of what you need to follow. Definitions:

1. entity errors (<entity>): a small part of a sentence, often an entity (e.g., location name), is incorrect (usually 1-3 words). Entity errors often involve noun phrases or nouns. If this happens, tag the wrong entity with <error>.
2. relational error (<relation>): a sentence is partially incorrect as a small part (usually 1 - 3 words). Relational errors often involve verbs and are often the opposite of what it should be. If this happens, tag the wrong relation with <error>.
3. contradictory sentence error (<contradictory>): a sentence where the entire sentence is contradicted by the given reference, meaning the sentence can be proven false due to a contradiction with information in the passage. if this happens, tag the wrong sentence with <error>.

Follow the given example exactly, your task is to create the edited completion with error tags <>:

##
Reference:
---
Marooned on Mars is a juvenile science fiction novel written by American writer Lester del Rey. It was published by John C. Winston Co. in 1952 with illustrations by Alex Schomburg.
---
Query:
---
Tell me about Marooned on Mars.
---
Passage:
---
Marooned on Mars is a science fiction novel aimed at a younger audience. It was written by Andy Weir and published by John C. Winston Co. in 1952, featuring illustrations by Alex Schomburg. It ended up having a readership of older boys despite efforts for it to be aimed at younger kids.
---
Output: 
---
Marooned on Mars is a science fiction novel aimed at a younger audience. It was written by <error>Andy Weir</error> and published by John C. Winston Co. in 1952, featuring illustrations by Alex Schomburg. <error>It ended up having a readership of older boys despite efforts for it to be aimed at younger kids.</error> 
---

Detect and tag errors in the following passage. Repeat the passage exactly, adding inline error tags where needed. Do not include any explanations, comments, or formatting symbols. If the passage has no errors, return it exactly as given, without any tags.

##
Reference: 
---
{doc}
---
Query:
---
{query}
---
Passage: 
---
{passage}
---
Output:
""".strip()

# def get_verification_prompt_fava(passage, doc):
#     return f"""
# Given a passage with factual errors, identify any <entity>, <relation>, <contradictory>, <subjective>, <unverifiable> or <invented> errors in the passage and add edits for <entity> and <relation> errors by inserting additional <mark></mark> or <delete></delete> tags to mark and delete. If there are no errors, return the passage with no tags. Any changes to the original passage should be marked in <> tags. Below are the error definitions followed by examples of what you need to follow. Definitions:

# 1. entity errors (<entity>): a small part of a sentence, often an entity (e.g., location name), is incorrect (usually 1-3 words). Entity errors often involve noun phrases or nouns.
# 2. relational error (<relation>): a sentence is partially incorrect as a small part (usually 1 - 3 words). Relational errors often involve verbs and are often the opposite of what it should be.
# 3. contradictory sentence error (<contradictory>): a sentence where the entire sentence is contradicted by the given reference, meaning the sentence can be proven false due to a contradiction with information in the passage.
# 4. invented info error (< invented >): these errors refer to entities that are not known or do not exist. This does not include fictional characters in books or movies. invented errors include phrases or sentences which have unknown entities or misleading information.
# 5. subjective sentence (<subjective>): an entire sentence or phrase that is subjective and cannot be verified, so it should not be included.
# 6. unverifiable sentence (<unverifiable>): a sentence where the whole sentence or phrase is unlikely to be factually grounded although it can be true, and the sentence cannot be confirmed nor denied using the reference given or internet search, it is often something personal or private and hence cannot be confirmed.

# Follow the given example exactly, your task is to create the edited completion with error tags <>:

# ##
# Reference:
# ---
# Marooned on Mars is a juvenile science fiction novel written by American writer Lester del Rey. It was published by John C. Winston Co. in 1952 with illustrations by Alex Schomburg.
# ---
# Passage:
# ---
# Marooned on Mars is a science fiction novel aimed at a younger audience. It was written by Andy Weir and published by John C. Winston Co. in 1952, featuring illustrations by Alex Schomburg. It ended up having a readership of older boys despite efforts for it to be aimed at younger kids. The novel inspired the famous Broadway musical "Stranded Stars," which won six Tony Awards. The novel tells a story of being stranded on the Purple Planet. I wish the novel had more exciting and thrilling plot twists.
# ---
# Edited: 
# ---
# Marooned on Mars is a science fiction novel aimed at a younger audience. It was written by <entity><mark>Lester del Rey</mark><delete>Andy Weir</delete></entity> and published by John C. Winston Co. in 1952, featuring illustrations by Alex Schomburg. <contradictory>It ended up having a readership of older boys despite efforts for it to be aimed at younger kids.</contradictory>. <invented>The novel inspired the famous Broadway musical "Stranded Stars," which won six Tony Awards.</invented> The novel tells a story of being stranded on the <entity><mark>Red</mark><delete>Purple</delete></entity> Planet. <subjective>I wish the novel had more exciting and thrilling plot twists.</subjective>
# ---

# ##
# Now detect errors and include edits in the following passage like done in the example above.
# Include error tags <> for ANYTHING YOU CHANGE IN THE ORIGINAL PASSAGE.

# Reference: 
# ---
# {doc}
# ---

# Passage: 
# ---
# {passage}
# ---

# Edited:
# """.strip()