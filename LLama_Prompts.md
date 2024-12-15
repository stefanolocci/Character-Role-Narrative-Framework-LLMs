# LLama-3.1-8B-Instruct Character-Role Narrative Framework Prompts

## Editorials

### Zero-Shot
Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Here’s how you should do it:

Look at each sentence: Go through the text sentence by sentence to find any characters. There can be no characters, one character, or many characters in a sentence.
Label the characters: When you find a character, give them the correct label (Hero, Villain, Victim, or Beneficiary) based only on what the text says. Do not guess or assume anything. Only label a character if the text clearly shows their role.
Pay attention to words: Words like "heal," "save," "suffer," "endangered," "protect," or similar words can help you decide the character's role. If these or similar words are not there, do not label based on your assumptions.
Roles can change: A character's role might change in the same sentence. Label them based on how they are described at that moment.

Use tags in the text: When you label a character, use the following tags directly in the text:
Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>

IMPORTANT: Rewrite the entire text with these tags included. Do not change the original text—just add the tags around the characters.

Remember, only label characters based on what is clearly stated in the text. Do not use your own knowledge or assumptions.

Here is the text to annotate:

### One-Shot
Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. There are only three types of entities that can be characters: human (including organizations or groups made of humans); instrumental (products like laws, measures, objects or processes); natural (natural elements, plants, animals, the planet itself and so on).
Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Here’s how you should do it:

Look at each sentence: Go through the text sentence by sentence to find any characters. There can be no characters, one character, or many characters in a sentence.

Label the characters: When you find a character, give them the correct label (Hero, Villain, Victim, or Beneficiary) based only on what the text says. Do not guess or assume anything. Only label a character if the text clearly shows their role. If there is no prominent character, do not label the text.

Use Linguistic Cues: Words like "heal," "save," "suffer," "endangered," "protect," or similar terms can indicate a character’s role. Additionally, consider the overall sentiment of the paragraph to help determine if an entity plays a role as a character.

Roles can change: A character's role might change in the same paragraph. Label them based on how they are described at that moment.

Use tags in the text: When you label a character, use the following tags directly in the text:

Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>

Consider the following as annotation example:

Input: \<anonymized\>

Output: \<anonymized\>

IMPORTANT: Rewrite the entire text with these tags included. Do not change the original text—just add the tags around the characters.

Remember, only label characters based on what is clearly stated in the text. Do not use your own knowledge or assumptions.

Here is the text to annotate:

### Few-Shot

Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. There are only three types of entities that can be characters: human (including organizations or groups made of humans); instrumental (products like laws, measures, objects or processes); natural (natural elements, plants, animals, the planet itself and so on).
Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Here’s how you should do it:

Look at each sentence: Go through the text sentence by sentence to find any characters. There can be no characters, one character, or many characters in a sentence.

Label the characters: When you find a character, give them the correct label (Hero, Villain, Victim, or Beneficiary) based only on what the text says. Do not guess or assume anything. Only label a character if the text clearly shows their role. If there is no prominent character, do not label the text.

Use Linguistic Cues: Words like "heal," "save," "suffer," "endangered," "protect," or similar terms can indicate a character’s role. Additionally, consider the overall sentiment of the paragraph to help determine if an entity plays a role as a character.

Roles can change: A character's role might change in the same paragraph. Label them based on how they are described at that moment.

Use tags in the text: When you label a character, use the following tags directly in the text:

Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>

Consider the following as annotation example:

Input: \<anonymized\>

Output: \<anonymized\>

Input: \<anonymized\>

Output: \<anonymized\>

Input: \<anonymized\>

Output: \<anonymized\>

IMPORTANT: Rewrite the entire text with these tags included. Do not change the original text—just add the tags around the characters.

Remember, only label characters based on what is clearly stated in the text. Do not use your own knowledge or assumptions.

Here is the text to annotate:

## Tweets

### Zero-Shot
Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Here’s how you should do it:

Look at each sentence: Go through the text sentence by sentence to find any characters. There can be no characters, one character, or many characters in a sentence.
Label the characters: When you find a character, give them the correct label (Hero, Villain, Victim, or Beneficiary) based only on what the text says. Do not guess or assume anything. Only label a character if the text clearly shows their role.
Pay attention to words: Words like "heal," "save," "suffer," "endangered," "protect," or similar words can help you decide the character's role. If these or similar words are not there, do not label based on your assumptions.
Roles can change: A character's role might change in the same sentence. Label them based on how they are described at that moment.

Use tags in the text: When you label a character, use the following tags directly in the text:
Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>

IMPORTANT: Rewrite the entire text with these tags included. Do not change the original text—just add the tags around the characters.

Remember, only label characters based on what is clearly stated in the text. Do not use your own knowledge or assumptions.

Here is the text to annotate:

### One-Shot
Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Here’s how you should do it:

Look at each sentence: Go through the text sentence by sentence to find any characters. There can be no characters, one character, or many characters in a sentence.
Label the characters: When you find a character, give them the correct label (Hero, Villain, Victim, or Beneficiary) based only on what the text says. Do not guess or assume anything. Only label a character if the text clearly shows their role.
Pay attention to words: Words like "heal," "save," "suffer," "endangered," "protect," or similar words can help you decide the character's role. If these or similar words are not there, do not label based on your assumptions.
Roles can change: A character's role might change in the same sentence. Label them based on how they are described at that moment.

Use tags in the text: When you label a character, use the following tags directly in the text:
Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>

For example, if the text is "The Government saved the environment," your output should be: "The <HER>Government</HER> saved the <BEN>environment</BEN>."

IMPORTANT: Rewrite the entire text with these tags included. Do not change the original text—just add the tags around the characters.

Remember, only label characters based on what is clearly stated in the text. Do not use your own knowledge or assumptions.

Here is the text to annotate:

### Few-Shot
Task Overview: You are given a text. Your task is to identify and label characters within the narrative. Characters are entities playing a clear role in the story, contributing to its core message. Label each identified character with one of the following roles:
Hero: Actively contributes to or endorses positive actions/events.
Villain: Responsible for negative actions or harm.
Victim: Suffers from or is endangered by actions/events, typically playing a passive role.
Beneficiary: Passively benefits from actions/events.
Character Types:
Human Characters: Humans or entities made up of people (e.g., corporations, governments, organizations).
Instrumental Characters: Abstract entities (e.g., policies, laws, technologies) produced by human characters that play a crucial role in the narrative.
Natural Characters: Non-human entities (e.g., animals, nature, natural processes) given agentive or passive roles within the narrative.
Instructions:
Identify Characters: Assess each sentence, sentence by sentence, to identify characters (there can be 0 to N characters per sentence).
Assign Roles: Label identified characters with the appropriate role based on how the narrative portrays them. Do not infer or imply any roles based on common knowledge or assumptions.  Only label characters if the text explicitly describes them in a way that fits a specific role (Hero, Villain, Victim, Beneficiary). Rely strictly on what is explicitly stated in the text—avoid making interpretations or assumptions.
Use Linguistic Indicators: Pay close attention to linguistic cues such as "heal," "save," "suffer from," "endangered by," "protect," and other similar phrases. These indicators will help determine the role of a character. If the text does not explicitly use such indicators or similar language, do not assign a role based on presumed implications.
Be Aware of Role Shifts: A character’s role can change as the sentence or paragraph progresses. Even if a character starts neutral, it might take on a role later in the sentence. Similarly, a character can switch roles within the same sentence or paragraph. Assign roles based on how the character is portrayed at each point in the text.
Focus on Narrative Perspective: Use linguistic indicators and context within the text to determine roles, strictly reflecting the author’s intended perspective. Avoid relying on external knowledge or common narratives—only label characters based on the explicit narrative context provided.
Label Nouns Only: Only label nouns or noun phrases, excluding articles (e.g., “the” in “the President”) and other parts of speech. Personal pronouns (e.g., “we,” “they”) can be labeled too.
Multiword Expressions: For multiword expressions (e.g., “President of the United States”), label the entire phrase, but avoid including unnecessary extensions.
Avoid Labeling Abstract Entities: Do not label overly abstract entities such as “decision.”
No Labeling If:
No clear narrative or characters/roles are identifiable.
The text is too short, vague, or the narrative is too implicit.
The text does not express the author’s perspective (e.g., reporting someone else’s perspective).
Output format: You must return the input text with each character labeled using in-line tag annotations (<start_token>text<end_token>), where the tag corresponds to a role name. The only available tags are:
Hero: <HER>text</HER>
Villain: <VIL>text</VIL>
Victim: <VIC>text</VIC>
Beneficiary: <BEN>text</BEN>
Consider the following as annotation examples:

Input: \<anonymized\>

Output: \<anonymized\>

Input: \<anonymized\>

Output: \<anonymized\>

Input: \<anonymized\>

Output: \<anonymized\>

Input: \<anonymized\>

Output: NO ANNOTATION

IMPORTANT: DO NOT CHANGE THE INPUT TEXT, ONLY ADD THE TAGS.
Note: Be attentive to the linguistic cues and specific wording used by the author, as they will guide you in assigning the correct roles. Avoid inferring roles based on outside knowledge or assumptions.

Here is the text to annotate:

## Character Categorization Prompt

Task Overview: You are given a text. Each text represents a unique entity, and your task is to categorize the entity based on one of three categories: Human, Instrumental, or Natural. Below are the definitions for each category, along with relevant examples.

Categories:
Human Characters:
These include humans or entities made up of people, such as corporations, governments, organizations of any type (e.g., religious), and political movements. Examples: 
"Oil and gas industry" (categorized as Human because it refers to a group of businesses).
"World" (categorized as Human when referring to governments, organizations, or companies).
"Low-income areas" (categorized as Human because it refers to the people living in those areas).
"Natural community" (categorized as Human when referring to a community of people living in harmony with nature).
Instrumental Characters:
These are more abstract entities such as policies, laws, technologies, measures, objects, or human-driven processes (e.g., "urbanization," "deforestation",  “city growth”). They can also be artifacts or processes that (i) have been produced or initiated by human characters
Examples:
"Pesticides and fertilizers" (categorized as Instrumental because they are human-made technologies).
"Carbon emissions" (categorized as Instrumental because they result from human processes).
"Rewilding" (categorized as Instrumental because it is a human-driven effort to help nature).
"Lower Snake River dams" (categorized as Instrumental because the dams are human-made structures).
"30x30 policy" (categorized as Instrumental because it refers to a human-created policy).
"Hunting" (categorized as Instrumental because it refers to a human-driven process).
"Buildings" or "Temples" (categorized as Instrumental because they are objects created by humans).
"PFAS" (categorized as Instrumental because it is a human-made technology/chemical).
Natural Characters: These comprise non-human entities such as natural elements (e.g., soil, oceans), animals, nature itself, and the planet. They can also include natural processes or phenomena (e.g., “biodiversity loss,” “climate change,” “pandemic”).
Examples:
"Europe" (categorized as Natural when referring to the geographical region and its natural elements, rather than its people).
"Smoke" (categorized as Natural when referring to poor air quality from smoke, assuming it is not human-caused).
Output Format:
You must return the input text with each entity labeled using in-line tag annotations (<start_token>text<end_token>), where the tag corresponds to a category name. The only available tags are:
Human: <HUM>text</HUM>
Instrumental: <INS>text</INS>
Natural: <NAT>text</NAT>
Examples:
<HUM>oil and gas industry</HUM>
<HUM>low-income communities</HUM>
<INS>30x30 policy</INS>
<INS>pesticides and fertilizers</INS>
<NAT>climate change</NAT>
<NAT>the ocean</NAT>

IMPORTANT: always consider the entity in input as a single one to annotate, even the ones with more than one word

Here is the text to annotate:

