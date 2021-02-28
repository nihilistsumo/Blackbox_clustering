from typing import List, Union

class InputTRECCARExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, qid: str = '', q_context: str = '', pids: List[str] = None, texts: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.qid = qid
        self.q_context = q_context
        self.pids = pids
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> \nqid: {}, \nqcontext: {}, \nlabel: {}, \npids: {}".format(str(self.qid), str(self.q_context), str(self.label), "; ".join(self.pids))