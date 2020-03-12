import collections
import numpy as np
import tensorflow.compat.v1 as tf

import ops
from config import config

MACCellTuple = collections.namedtuple("MACCellTuple", ("control", "memory"))

'''
The MAC cell.

Recurrent cell for multi-step reasoning. Presented in https://arxiv.org/abs/1803.03067.
The cell has recurrent control and memory states that interact with the question 
and knowledge base (image) respectively.

The hidden state structure is MACCellTuple(control, memory)

At each step the cell performs by calling to three subunits: control, read and write.

1. The Control Unit computes the control state by computing attention over the question words.
The control state represents the current reasoning operation the cell performs.

2. The Read Unit retrieves information from the knowledge base, given the control and previous
memory values, by computing 2-stages attention over the knowledge base.

3. The Write Unit integrates the retrieved information to the previous hidden memory state,
given the value of the control state, to perform the current reasoning operation.
'''


class MACCell:
    '''Initialize the MAC cell.
    (Note that in the current version the cell is stateful --
    updating its own internals when being called)

    Args:
        question_embeddings: the vector representation of the questions.
        [batch_size, ctrlDim]

        question_word_embeddings: the question words embeddings.
        [batch_size, questionLength, ctrlDim]

        question_contextual_word_embeddings: the encoder outputs -- the "contextual" question words.
        [batch_size, questionLength, ctrlDim]

        question_lengths: the length of each question.
        [batch_size]

        memoryDropout: dropout on the memory state (Tensor scalar).
        readDropout: dropout inside the read unit (Tensor scalar).
        writeDropout: dropout on the new information that gets into the write unit (Tensor scalar).

        batch_size: batch size (Tensor scalar).
        train: train or test mod (Tensor boolean).
        reuse: reuse cell

        knowledge_base:
    '''

    def __init__(
            self,
            params,
            placeholders,
            question_embeddings,
            question_word_embeddings,
            question_contextual_word_embeddings,
            question_lengths,
            knowledge_base,
            dropout,
            batch_size,
            reuse=None):
        self.params = params
        self.configs = params.model.mac
        self.placeholders = placeholders

        self.question_embeddings = question_embeddings
        self.question_word_embeddings = question_word_embeddings
        self.question_contextual_word_embeddings = question_contextual_word_embeddings
        self.question_lengths = question_lengths

        self.knowledge_base = knowledge_base
        self.dropout = dropout
        self.batch_size = batch_size
        self.reuse = reuse

    ''' 
    Cell state size. 
    '''

    @property
    def state_size(self):
        return MACCellTuple(config.ctrlDim, config.memDim)

    '''
    Cell output size. Currently it doesn't have any outputs. 
    '''

    @property
    def output_size(self):
        return 1

    # pass encoder hidden states to control?
    '''
    The Control Unit: computes the new control state -- the reasoning operation,
    by summing up the word embeddings according to a computed attention distribution.

    The unit is recurrent: it receives the whole question and the previous control state,
    merge them together (resulting in the "continuous control"), and then uses that 
    to compute attentions over the question words. Finally, it combines the words 
    together according to the attention distribution to get the new control state. 

    Args:
        controlInput: external inputs to control unit (the question vector).
        [batch_size, ctrlDim]

        inWords: the representation of the words used to compute the attention.
        [batch_size, questionLength, ctrlDim]

        outWords: the representation of the words that are summed up. 
                  (by default inWords == outWords)
        [batch_size, questionLength, ctrlDim]

        question_lengths: the length of each question.
        [batch_size]

        control: the previous control hidden state value.
        [batch_size, ctrlDim]

        contControl: optional corresponding continuous control state
        (before casting the attention over the words).
        [batch_size, ctrlDim]

    Returns:
        the new control state
        [batch_size, ctrlDim]

        the continuous (pre-attention) control
        [batch_size, ctrlDim]
    '''

    def control(self, controlInput, inWords, outWords, question_lengths,
                control, contControl=None, name="", reuse=None):

        with tf.variable_scope("control" + name, reuse=reuse):
            dim = config.ctrlDim

            ## Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = controlInput
            if config.controlFeedPrev:
                newContControl = control if config.controlFeedPrevAtt else contControl
                if config.controlFeedInputs:
                    newContControl = tf.concat([newContControl, controlInput], axis=-1)
                    dim += config.ctrlDim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, config.ctrlDim,
                                            act=config.controlContAct, name="contControl")
                dim = config.ctrlDim

            ## Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis=1) * inWords

            # optionally concatenate words
            if config.controlConcatWords:
                interactions = tf.concat([interactions, inWords], axis=-1)
                dim += config.ctrlDim

                # optional projection
            if config.controlProj:
                interactions = ops.linear(interactions, dim, config.ctrlDim,
                                          act=config.controlProjAct)
                dim = config.ctrlDim

            # compute attention distribution over words and summarize them accordingly
            logits = ops.inter2logits(interactions, dim)
            # self.interL = (interW, interb)

            # if config.controlCoverage:
            #     logits += coverageBias * coverage

            attention = tf.nn.softmax(ops.expMask(logits, question_lengths))
            self.attentions["question"].append(attention)

            # if config.controlCoverage:
            #     coverage += attention # Add logits instead?

            newControl = ops.att2Smry(attention, outWords)

            # ablation: use continuous control (pre-attention) instead
            if config.controlContinuous:
                newControl = newContControl

        return newControl, newContControl

    '''
    The read unit extracts relevant information from the knowledge base given the
    cell's memory and control states. It computes attention distribution over
    the knowledge base by comparing it first to the memory and then to the control.
    Finally, it uses the attention distribution to sum up the knowledge base accordingly,
    resulting in an extraction of relevant information. 

    Args:
        knowledge base: representation of the knowledge base (image). 
        [batch_size, kbSize (Height * Width), memDim]

        memory: the cell's memory state
        [batch_size, memDim]

        control: the cell's control state
        [batch_size, ctrlDim]

    Returns the information extracted.
    [batch_size, memDim]
    '''

    def read(self, knowledge_base, memory, control, name="", reuse=None):
        with tf.variable_scope("read" + name, reuse=reuse):
            dim = config.memDim

            ## memory dropout
            if config.memoryVariationalDropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropout)
            else:
                memory = tf.nn.dropout(memory, self.dropout)

            ## Step 1: knowledge base / memory interactions
            # parameters for knowledge base and memory projection
            proj = None
            if config.readProjInputs:
                proj = {"dim": config.attDim, "shared": config.readProjShared, "dropout": self.dropout}
                dim = config.attDim

            # parameters for concatenating knowledge base elements
            concat = {"x": config.readMemConcatKB, "proj": config.readMemConcatProj}

            # compute interactions between knowledge base and memory
            interactions, interDim = ops.mul(x=knowledge_base, y=memory, dim=config.memDim,
                                             proj=proj, concat=concat, interMod=config.readMemAttType, name="memInter")

            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if config.readMemProj:
                interactions = ops.linear(interactions, interDim, dim, act=config.readMemAct,
                                          name="memKbProj")
            else:
                dim = interDim

            ## Step 2: compute interactions with control
            if config.readCtrl:
                # compute interactions with control
                if config.ctrlDim != dim:
                    control = ops.linear(control, ctrlDim, dim, name="ctrlProj")

                interactions, interDim = ops.mul(interactions, control, dim,
                                                 interMod=config.readCtrlAttType,
                                                 concat={"x": config.readCtrlConcatInter},
                                                 name="ctrlInter")

                # optionally concatenate knowledge base elements
                if config.readCtrlConcatKB:
                    if config.readCtrlConcatProj:
                        addedInp, addedDim = projectedKB, config.attDim
                    else:
                        addedInp, addedDim = knowledge_base, config.memDim
                    interactions = tf.concat([interactions, addedInp], axis=-1)
                    dim += addedDim

                    # optional nonlinearity
                interactions = ops.activations[config.readCtrlAct](interactions)

            ## Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            attention = ops.inter2att(interactions, dim, dropout=self.dropout)

            self.attentions["kb"].append(attention)

            # optionally use projected knowledge base instead of original
            if config.readSmryKBProj:
                knowledge_base = projectedKB

            # sum up the knowledge base according to the distribution
            information = ops.att2Smry(attention, knowledge_base)

        return information

    '''
    The write unit integrates newly retrieved information (from the read unit),
    with the cell's previous memory hidden state, resulting in a new memory value.
    The unit optionally supports:
    1. Self-attention to previous control / memory states, in order to consider previous steps
    in the reasoning process.
    2. Gating between the new memory and previous memory states, to allow dynamic adjustment
    of the reasoning process length.

    Args:
        memory: the cell's memory state
        [batch_size, memDim]

        info: the information to integrate with the memory
        [batch_size, memDim]

        control: the cell's control state
        [batch_size, ctrlDim]

        contControl: optional corresponding continuous control state 
        (before casting the attention over the words).
        [batch_size, ctrlDim]

    Return the new memory 
    [batch_size, memDim]
    '''

    def write(self, memory, info, control, contControl=None, name="", reuse=None):
        with tf.variable_scope("write" + name, reuse=reuse):

            # optionally project info
            if config.writeInfoProj:
                info = ops.linear(info, config.memDim, config.memDim, name="info")

            # optional info nonlinearity
            info = ops.activations[config.writeInfoAct](info)

            # compute self-attention vector based on previous controls and memories
            if config.writeSelfAtt:
                selfControl = control
                if config.writeSelfAttMod == "CONT":
                    selfControl = contControl
                # elif config.writeSelfAttMod == "POST":
                #     selfControl = postControl
                selfControl = ops.linear(selfControl, config.ctrlDim, config.ctrlDim, name="ctrlProj")

                interactions = self.controls * tf.expand_dims(selfControl, axis=1)

                # if config.selfAttShareInter:
                #     selfAttlogits = self.linearP(selfAttInter, config.encDim, 1, self.interL[0], self.interL[1], name = "modSelfAttInter")
                attention = ops.inter2att(interactions, config.ctrlDim, name="selfAttention")
                self.attentions["self"].append(attention)
                selfSmry = ops.att2Smry(attention, self.memories)

            # get write unit inputs: previous memory, the new info, optionally self-attention / control
            newMemory, dim = memory, config.memDim
            if config.writeInputs == "INFO":
                newMemory = info
            elif config.writeInputs == "SUM":
                newMemory += info
            elif config.writeInputs == "BOTH":
                newMemory, dim = ops.concat(newMemory, info, dim, mul=config.writeConcatMul)
            # else: MEM

            if config.writeSelfAtt:
                newMemory = tf.concat([newMemory, selfSmry], axis=-1)
                dim += config.memDim

            if config.writeMergeCtrl:
                newMemory = tf.concat([newMemory, control], axis=-1)
                dim += config.memDim

                # project memory back to memory dimension
            if config.writeMemProj or (dim != config.memDim):
                newMemory = ops.linear(newMemory, dim, config.memDim, name="newMemory")

            # optional memory nonlinearity
            newMemory = ops.activations[config.writeMemAct](newMemory)

            # write unit gate
            if config.writeGate:
                gateDim = config.memDim
                if config.writeGateShared:
                    gateDim = 1

                z = tf.sigmoid(ops.linear(control, config.ctrlDim, gateDim, name="gate", bias=config.writeGateBias))

                self.attentions["gate"].append(z)

                newMemory = newMemory * z + memory * (1 - z)

                # optional batch normalization
            if config.memoryBN:
                newMemory = tf.contrib.layers.batch_norm(
                    newMemory, decay=config.bnDecay,
                    center=config.bnCenter, scale=config.bnScale,
                    is_training=self.placeholders.is_training,
                    updates_collections=None)

        return newMemory

    def memAutoEnc(self, newMemory, info, control, name="", reuse=None):
        with tf.variable_scope("memAutoEnc" + name, reuse=reuse):
            # inputs to auto encoder
            features = info if config.autoEncMemInputs == "INFO" else newMemory
            features = ops.linear(features, config.memDim, config.ctrlDim,
                                  act=config.autoEncMemAct, name="aeMem")

            # reconstruct control
            if config.autoEncMemLoss == "CONT":
                loss = tf.reduce_mean(tf.squared_difference(control, features))
            else:
                interactions, dim = ops.mul(self.question_contextual_word_embeddings, features, config.ctrlDim,
                                            concat={"x": config.autoEncMemCnct}, mulBias=config.mulBias, name="aeMem")

                logits = ops.inter2logits(interactions, dim)
                logits = self.expMask(logits, self.question_lengths)

                # reconstruct word attentions
                if config.autoEncMemLoss == "PROB":
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.attentions["question"][-1], logits=logits))

                # reconstruct control through words attentions
                else:
                    attention = tf.nn.softmax(logits)
                    summary = ops.att2Smry(attention, self.question_contextual_word_embeddings)
                    loss = tf.reduce_mean(tf.squared_difference(control, summary))

        return loss

    '''
    Call the cell to get new control and memory states.

    Args:
        inputs: in the current implementation the cell don't get recurrent inputs
        every iteration (argument for comparability with rnn interface).

        state: the cell current state (control, memory)
        MACCellTuple([batch_size, ctrlDim],[batch_size, memDim])

    Returns the new state -- the new memory and control values.
    MACCellTuple([batch_size, ctrlDim],[batch_size, memDim])
    '''

    def __call__(self, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse=self.reuse):  # as tfscope
            control = state.control
            memory = state.memory

            # cell sharing
            inputName = "qInput"
            inputNameU = "qInputU"
            inputReuseU = inputReuse = (self.iteration > 0)
            if config.controlInputUnshared:
                inputNameU = "qInput%d" % self.iteration
                inputReuseU = None

            cellName = ""
            cellReuse = (self.iteration > 0)
            if config.unsharedCells:
                cellName = str(self.iteration)
                cellReuse = None

                ## control unit
            # prepare question input to control
            controlInput = ops.linear(self.question_embeddings, config.ctrlDim, config.ctrlDim,
                                      name=inputName, reuse=inputReuse)

            controlInput = ops.activations[config.controlInputAct](controlInput)

            controlInput = ops.linear(controlInput, config.ctrlDim, config.ctrlDim,
                                      name=inputNameU, reuse=inputReuseU)

            newControl, self.contControl = self.control(controlInput, self.inWords, self.outWords,
                                                        self.question_lengths, control, self.contControl, name=cellName,
                                                        reuse=cellReuse)

            # read unit
            # ablation: use whole question as control
            if config.controlWholeQ:
                newControl = self.question_embeddings
                # ops.linear(self.question_embeddings, config.ctrlDim, projDim, name = "qMod")

            info = self.read(self.knowledge_base, memory, newControl, name=cellName, reuse=cellReuse)

            if config.writeDropout < 1.0:
                # write unit
                info = tf.nn.dropout(info, self.dropout)

            newMemory = self.write(memory, info, newControl, self.contControl, name=cellName, reuse=cellReuse)

            # add auto encoder loss for memory
            # if config.autoEncMem:
            #     self.autoEncLosses["memory"] += memAutoEnc(newMemory, info, newControl)

            # append as standard list?
            self.controls = tf.concat([self.controls, tf.expand_dims(newControl, axis=1)], axis=1)
            self.memories = tf.concat([self.memories, tf.expand_dims(newMemory, axis=1)], axis=1)
            self.infos = tf.concat([self.infos, tf.expand_dims(info, axis=1)], axis=1)

            # self.contControls = tf.concat([self.contControls, tf.expand_dims(contControl, axis = 1)], axis = 1)
            # self.postControls = tf.concat([self.controls, tf.expand_dims(postControls, axis = 1)], axis = 1)

        newState = MACCellTuple(newControl, newMemory)
        return newState

    '''
    Initializes the a hidden state to based on the value of the initType:
    "PRM" for parametric initialization
    "ZERO" for zero initialization  
    "Q" to initialize to question vectors.

    Args:
        name: the state variable name.
        dim: the dimension of the state.
        initType: the type of the initialization
        batch_size: the batch size

    Returns the initialized hidden state.
    '''

    def initState(self, name, dim, initType, batch_size):
        if initType == "PRM":
            prm = tf.get_variable(name, shape=(dim,),
                                  initializer=tf.random_normal_initializer())
            initState = tf.tile(tf.expand_dims(prm, axis=0), [batch_size, 1])
        elif initType == "ZERO":
            initState = tf.zeros((batch_size, dim), dtype=tf.float32)
        else:  # "Q"
            initState = self.question_embeddings
        return initState

    '''
    Add a parametric null word to the questions.

    Args:
        words: the words to add a null word to.
        [batch_size, questionLentgth]

        lengths: question lengths.
        [batch_size] 

    Returns the updated word sequence and lengths.  
    '''

    def addNullWord(words, lengths):
        nullWord = tf.get_variable("zeroWord", shape=(1, config.ctrlDim), initializer=tf.random_normal_initializer())
        nullWord = tf.tile(tf.expand_dims(nullWord, axis=0), [self.batch_size, 1, 1])
        words = tf.concat([nullWord, words], axis=1)
        lengths += 1
        return words, lengths

    '''
    Initializes the cell internal state (currently it's stateful). In particular,
    1. Data-structures (lists of attention maps and accumulated losses).
    2. The memory and control states.
    3. The knowledge base (optionally merging it with the question vectors)
    4. The question words used by the cell (either the original word embeddings, or the 
       encoder outputs, with optional projection).

    Args:
        batch_size: the batch size

    Returns the initial cell state.
    '''

    def zero_state(self, batch_size, dtype=tf.float32):
        ## initialize data-structures
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}

        ## initialize state
        initialControl = self.initState("initCtrl", config.ctrlDim, config.initCtrl, batch_size)
        initialMemory = self.initState("initMem", config.memDim, config.initMem, batch_size)

        self.controls = tf.expand_dims(initialControl, axis=1)
        self.memories = tf.expand_dims(initialMemory, axis=1)
        self.infos = tf.expand_dims(initialMemory, axis=1)

        self.contControl = initialControl
        # self.contControls = tf.expand_dims(initialControl, axis = 1)
        # self.postControls = tf.expand_dims(initialControl, axis = 1)

        ## initialize knowledge base
        # optionally merge question into knowledge base representation
        if config.initKBwithQ != "NON":
            iVecQuestions = ops.linear(self.question_embeddings, config.ctrlDim, config.memDim, name="questions")

            concatMul = (config.initKBwithQ == "MUL")
            cnct, dim = ops.concat(self.knowledge_base, iVecQuestions, config.memDim, mul=concatMul, expandY=True)
            self.knowledge_base = ops.linear(cnct, dim, config.memDim, name="initKB")

        ## initialize question words
        # choose question words to work with (original embeddings or encoder outputs)
        words = self.question_contextual_word_embeddings if config.controlContextual else self.question_word_embeddings

        # optionally add parametric "null" word in the to all questions
        if config.addNullWord:
            words, question_lengths = self.addNullWord(words, question_lengths)

        # project words
        self.inWords = self.outWords = words
        if config.controlInWordsProj or config.controlOutWordsProj:
            pWords = ops.linear(words, config.ctrlDim, config.ctrlDim, name="wordsProj")
            self.inWords = pWords if config.controlInWordsProj else words
            self.outWords = pWords if config.controlOutWordsProj else words

        # if config.controlCoverage:
        #     self.coverage = tf.zeros((batch_size, tf.shape(words)[1]), dtype = tf.float32)
        #     self.coverageBias = tf.get_variable("coverageBias", shape = (),
        #         initializer = config.controlCoverageBias)

        ## initialize memory variational dropout mask
        if config.memoryVariationalDropout:
            self.memDpMask = ops.generateVarDpMask((batch_size, config.memDim), self.dropout)

        return MACCellTuple(initialControl, initialMemory)
