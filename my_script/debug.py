prompt_template= "a young woman with a {}, wearing a pink shirt. She is standing in front of a fence, possibly in a park or an outdoor setting. The woman appears to be enjoying her time outdoors, possibly engaging in a sport or a recreational activity. "
expression_leys = ['bright smile', 'sad face', 'astonished face', 'exaggerated expression']
for expression_ley in expression_leys:
    print(prompt_template.format(expression_ley))
    print('\n')