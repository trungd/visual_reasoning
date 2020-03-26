from collections import namedtuple

batch_x_fields = (
    "images",
    "questions",
    "question_lengths",
    "question_bert_outputs",
    "question_bert_states",
    "question_bert_lengths",
    "objects",
    "object_lengths",
    "object_bounding_boxes")
BatchX = namedtuple("BatchX", batch_x_fields, defaults=(None,) * len(batch_x_fields))
