description_di_di_each_label = {
'related to': 'The diseases are related because they belong to the same disease, category, type, or are manifestations of the same condition.',
'comorbidity': 'Distinct diseases that explicitly occur together. These diseases are typically independent but coexist.',
'no relation': 'The diseases have no relationship, either because one of the entities is not a disease, or because they do not share any clinical, pathological, or causative links.',
'other': 'This category includes any relationship between diseases that does not fall under the other specified labels.'
}

description_dr_dr_each_label = {
'related to': 'The drugs are related because they belong to the same drug',
'combined with': 'The drugs are explicit combined for the same disease, based on the context of the text.',
'replaced by': 'The head drug is replaced by the tail drug, based on the context of the text.',
'no relation': 'The drugs have no relationship, either because one of the entities is not a drug, or because they do not share any clinical, pathological, or causative links.',
'other': 'This category includes any relationship between drugs that does not fall under the other specified labels.'
}

description_di_dr_each_label = {
'indicated': 'The drug in question is administered for the disease in question, based on the context of the text.',
'no_indicated': 'The drug in question is not indicated for the disease in question, based on the context of the text.',
'discontinued': 'The drug in question was previously used but has been stopped for the disease in question, based on the context of the text.',
'no relation': 'The drug in question has explicitly no known relationship with the disease in this context.',
'other': 'The relationship between the drug and disease is not clearly described by the other categories in this context.'
}
