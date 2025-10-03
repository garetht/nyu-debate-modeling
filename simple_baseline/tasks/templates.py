MULTIPLE_CHOICE_TEMPLATE = r"""
{question}

Answer the preceding multiple choice question about the information above. 
The entire content of your response should absolutely and without exception be of the 
following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{choices}
"""
