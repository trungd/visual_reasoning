import collections
from typing import Any

from . import ops
import tensorflow.compat.v1 as tf
from dlex import Params, Dict
from dlex.datasets.tf import Dataset
from dlex.tf.models.base_v1 import BaseModelV1
from dlex.tf.utils.tf_v1 import get_activation_fn, multi_layer_cnn

MACCellTuple = collections.namedtuple("MACCellTuple", ("control", "memory"))
Placeholders = collections.namedtuple(
    "Placeholders",
    "images questions question_lengths answers dropout")

'''
The MAC network model. It performs reasoning processes to answer a question over
knowledge base (the image) by decomposing it into attention-based computational steps,
each perform by a recurrent MAC cell.

The network has three main components. 
Input unit: processes the network inputs: raw question strings and image into
distributional representations.

The MAC network: calls the MACcells (mac_cell.py) config.netLength number of times,
to perform the reasoning process over the question and image.

The output unit: a classifier that receives the question and final state of the MAC
network and uses them to compute log-likelihood over the possible one-word answers.       
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
            params: Params,
            placeholders: Placeholders,
            question_embeddings,
            question_word_embeddings,
            question_contextual_word_embeddings,
            question_lengths,
            knowledge_base,
            reuse=None):
        self.params = params
        self.configs = params.model.mac
        self.placeholders = placeholders

        self.question_embeddings = question_embeddings
        self.question_word_embeddings = question_word_embeddings
        self.question_contextual_word_embeddings = question_contextual_word_embeddings
        self.question_lengths = question_lengths

        self.knowledge_base = knowledge_base
        self.dropout = placeholders.dropout
        self.reuse = reuse
        
        self.control_dim = params.model.control_dim
        self.memory_dim = params.model.memory_dim

    ''' 
    Cell state size. 
    '''

    @property
    def state_size(self):
        return MACCellTuple(self.control_dim, self.memory_dim)

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
        control_input: external inputs to control unit (the question vector).
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

    def control(self, control_input, inWords, outWords, question_lengths,
                control, contControl=None, name="", reuse=None):
        cfg = self.configs.control
        with tf.variable_scope("control" + name, reuse=reuse):
            dim = self.control_dim

            ## Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = control_input
            if cfg.controlFeedPrev:
                newContControl = control if cfg.controlFeedPrevAtt else contControl
                if cfg.controlFeedInputs:
                    newContControl = tf.concat([newContControl, control_input], axis=-1)
                    dim += self.control_dim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, self.control_dim,
                                            act=cfg.controlContAct, name="contControl")
                dim = self.control_dim

            ## Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis=1) * inWords

            # optionally concatenate words
            if cfg.concat_words:
                interactions = tf.concat([interactions, inWords], axis=-1)
                dim += self.control_dim

                # optional projection
            if cfg.project:
                interactions = ops.linear(interactions, dim, self.control_dim, act=cfg.controlProjAct)
                dim = self.control_dim

            # compute attention distribution over words and summarize them accordingly
            logits = ops.inter2logits(interactions, dim)
            # self.interL = (interW, interb)

            attention = tf.nn.softmax(ops.expMask(logits, question_lengths))
            self.attentions["question"].append(attention)

            new_control = ops.att2Smry(attention, outWords)

            # ablation: use continuous control (pre-attention) instead
            if cfg.controlContinuous:
                new_control = newContControl

        return new_control, newContControl

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
        cfg = self.configs.read
        with tf.variable_scope("read" + name, reuse=reuse):
            dim = self.memory_dim

            ## memory dropout
            if self.configs.memory_variational_dropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropout)
            else:
                memory = tf.nn.dropout(memory, rate=self.dropout)

            ## Step 1: knowledge base / memory interactions
            # parameters for knowledge base and memory projection
            proj = None
            if cfg.project_inputs:
                proj = {"dim": cfg.attention_dim, "shared": cfg.readProjShared, "dropout": self.dropout}
                dim = cfg.attention_dim

            # parameters for concatenating knowledge base elements
            concat = {"x": cfg.memory_concat_knowledge_base, "proj": cfg.memory_concat_knowledge_base}

            # compute interactions between knowledge base and memory
            interactions, interDim = ops.mul(
                x=knowledge_base,
                y=memory,
                dim=self.memory_dim,
                proj=proj,
                concat=concat,
                interMod=cfg.attention_type,
                name="memInter")

            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if cfg.project_memory:
                interactions = ops.linear(
                    interactions, interDim, dim, act=cfg.activation_fn,
                    name="memKbProj")
            else:
                dim = interDim

            ## Step 2: compute interactions with control
            if cfg.control:
                # compute interactions with control
                if self.control_dim != dim:
                    control = ops.linear(control, self.control_dim, dim, name="ctrlProj")

                interactions, interDim = ops.mul(
                    interactions, control, dim,
                    interMod=cfg.control_attention_type,
                    concat={"x": cfg.control_concat_interactions},
                    name="ctrlInter")

                # optionally concatenate knowledge base elements
                if cfg.control_concat_knowledge_base:
                    if cfg.control_concat_projection:
                        addedInp, addedDim = projectedKB, cfg.attDim
                    else:
                        addedInp, addedDim = knowledge_base, self.memory_dim
                    interactions = tf.concat([interactions, addedInp], axis=-1)
                    dim += addedDim

                    # optional nonlinearity
                interactions = get_activation_fn(cfg.activation_fn)(interactions)

            ## Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            attention = ops.inter2att(interactions, dim, dropout=self.dropout)

            self.attentions["kb"].append(attention)

            # optionally use projected knowledge base instead of original
            if cfg.readSmryKBProj:
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
        cfg = self.configs.write
        with tf.variable_scope("write" + name, reuse=reuse):
            # optionally project info
            if cfg.project_info:
                info = ops.linear(info, self.memory_dim, self.memory_dim, name="info")

            # optional info nonlinearity
            info = get_activation_fn(cfg.info_activation_fn)(info)

            # compute self-attention vector based on previous controls and memories
            if cfg.self_attention:
                selfControl = control
                if cfg.writeSelfAttMod == "CONT":
                    selfControl = contControl
                # elif cfg.writeSelfAttMod == "POST":
                #     selfControl = postControl
                selfControl = ops.linear(selfControl, self.control_dim, self.control_dim, name="ctrlProj")

                interactions = self.controls * tf.expand_dims(selfControl, axis=1)

                # if cfg.selfAttShareInter:
                #     selfAttlogits = self.linearP(selfAttInter, cfg.encDim, 1, self.interL[0], self.interL[1], name = "modSelfAttInter")
                attention = ops.inter2att(interactions, self.control_dim, name="selfAttention")
                self.attentions["self"].append(attention)
                selfSmry = ops.att2Smry(attention, self.memories)

            # get write unit inputs: previous memory, the new info, optionally self-attention / control
            new_memory, dim = memory, self.memory_dim
            if cfg.inputs == "info":
                new_memory = info
            elif cfg.inputs == "sum":
                new_memory += info
            elif cfg.inputs == "info_sum":
                new_memory, dim = ops.concat(new_memory, info, dim, mul=cfg.writeConcatMul)
            # else: MEM

            if cfg.self_attention:
                new_memory = tf.concat([new_memory, selfSmry], axis=-1)
                dim += self.memory_dim

            if cfg.merge_control:
                new_memory = tf.concat([new_memory, control], axis=-1)
                dim += self.memory_dim

                # project memory back to memory dimension
            if cfg.project_memory or (dim != self.memory_dim):
                new_memory = ops.linear(new_memory, dim, self.memory_dim, name="new_memory")

            # optional memory nonlinearity
            new_memory = get_activation_fn(cfg.memory_activation_fn)(new_memory)

            # write unit gate
            if cfg.memory_gate:
                gateDim = self.memory_dim
                if cfg.writeGateShared:
                    gateDim = 1

                z = tf.sigmoid(ops.linear(control, self.control_dim, gateDim, name="gate", bias=cfg.writeGateBias))

                self.attentions["gate"].append(z)

                new_memory = new_memory * z + memory * (1 - z)

                # optional batch normalization
            if cfg.memory_batch_norm:
                new_memory = tf.contrib.layers.batch_norm(
                    new_memory, decay=cfg.bnDecay,
                    center=cfg.bnCenter, scale=cfg.bnScale,
                    is_training=self.is_training,
                    updates_collections=None)

        return new_memory

    def memAutoEnc(self, new_memory, info, control, name="", reuse=None):
        with tf.variable_scope("memAutoEnc" + name, reuse=reuse):
            # inputs to auto encoder
            features = info if cfg.autoEncMemInputs == "INFO" else new_memory
            features = ops.linear(features, self.memory_dim, self.control_dim,
                                  act=cfg.autoEncMemAct, name="aeMem")

            # reconstruct control
            if cfg.autoEncMemLoss == "CONT":
                loss = tf.reduce_mean(tf.squared_difference(control, features))
            else:
                interactions, dim = ops.mul(self.question_contextual_word_embeddings, features, self.control_dim,
                                            concat={"x": cfg.autoEncMemCnct}, mulBias=cfg.mulBias, name="aeMem")

                logits = ops.inter2logits(interactions, dim)
                logits = self.expMask(logits, self.question_lengths)

                # reconstruct word attentions
                if cfg.autoEncMemLoss == "PROB":
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
        cfg = self.configs
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse=self.reuse):  # as tfscope
            control = state.control
            memory = state.memory

            # cell sharing
            inputName = "qInput"
            inputNameU = "qInputU"
            inputReuseU = inputReuse = (self.iteration > 0)
            if not cfg.control.share_input_layer:
                inputNameU = "qInput%d" % self.iteration
                inputReuseU = None

            cellName = ""
            cellReuse = (self.iteration > 0)
            if cfg.unsharedCells:
                cellName = str(self.iteration)
                cellReuse = None

            # control unit
            # prepare question input to control
            control_input = ops.linear(
                self.question_embeddings, self.control_dim, self.control_dim,
                name=inputName, reuse=inputReuse)

            control_input = get_activation_fn(cfg.control.input_activation_fn)(control_input)

            control_input = ops.linear(
                control_input, self.control_dim, self.control_dim,
                name=inputNameU, reuse=inputReuseU)

            new_control, self.contControl = self.control(
                control_input, self.inWords, self.outWords,
                self.question_lengths, control, self.contControl, name=cellName, reuse=cellReuse)

            info = self.read(self.knowledge_base, memory, new_control, name=cellName, reuse=cellReuse)
            info = tf.nn.dropout(info, rate=self.dropout)
            new_memory = self.write(memory, info, new_control, self.contControl, name=cellName, reuse=cellReuse)

            # add auto encoder loss for memory
            # if cfg.autoEncMem:
            #     self.autoEncLosses["memory"] += memAutoEnc(new_memory, info, new_control)

            # append as standard list?
            self.controls = tf.concat([self.controls, tf.expand_dims(new_control, axis=1)], axis=1)
            self.memories = tf.concat([self.memories, tf.expand_dims(new_memory, axis=1)], axis=1)
            self.infos = tf.concat([self.infos, tf.expand_dims(info, axis=1)], axis=1)

            # self.contControls = tf.concat([self.contControls, tf.expand_dims(contControl, axis = 1)], axis = 1)
            # self.postControls = tf.concat([self.controls, tf.expand_dims(postControls, axis = 1)], axis = 1)

        newState = MACCellTuple(new_control, new_memory)
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

    def zero_state(self, batch_size):
        cfg = self.configs
        ## initialize data-structures
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}

        ## initialize state
        initialControl = self.initState("initCtrl", self.control_dim, cfg.control_init, batch_size)
        initialMemory = self.initState("initMem", self.memory_dim, cfg.memory_init, batch_size)

        self.controls = tf.expand_dims(initialControl, axis=1)
        self.memories = tf.expand_dims(initialMemory, axis=1)
        self.infos = tf.expand_dims(initialMemory, axis=1)

        self.contControl = initialControl
        # self.contControls = tf.expand_dims(initialControl, axis = 1)
        # self.postControls = tf.expand_dims(initialControl, axis = 1)

        ## initialize knowledge base
        # optionally merge question into knowledge base representation
        if cfg.initKBwithQ is not None:
            iVecQuestions = ops.linear(self.question_embeddings, self.control_dim, self.memory_dim, name="questions")

            concatMul = (cfg.initKBwithQ == "MUL")
            cnct, dim = ops.concat(self.knowledge_base, iVecQuestions, self.memory_dim, mul=concatMul, expandY=True)
            self.knowledge_base = ops.linear(cnct, dim, self.memory_dim, name="initKB")

        # initialize question words
        # choose question words to work with (original embeddings or encoder outputs)
        words = self.question_contextual_word_embeddings

        # project words
        self.inWords = self.outWords = words
        if cfg.controlInWordsProj or cfg.controlOutWordsProj:
            pWords = ops.linear(words, self.control_dim, self.control_dim, name="wordsProj")
            self.inWords = pWords if cfg.controlInWordsProj else words
            self.outWords = pWords if cfg.controlOutWordsProj else words

        # initialize memory variational dropout mask
        if self.configs.memory_variational_dropout:
            self.memDpMask = ops.generateVarDpMask((batch_size, self.memory_dim), self.dropout)

        return MACCellTuple(initialControl, initialMemory)


class MAC(BaseModelV1):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)
        with tf.variable_scope("placeholders"):
            self._placeholders = Placeholders(
                images=tf.placeholder(tf.float32, shape=(
                    None,
                    params.model.image.output_num_channels,
                    params.model.image.output_height,
                    params.model.image.output_width)),
                questions=tf.placeholder(tf.int32, shape=(None, None)),
                question_lengths=tf.placeholder(tf.int32, shape=(None,)),
                answers=tf.placeholder(tf.int32, shape=(None,)),
                dropout=tf.placeholder(tf.float32, shape=(), name="dropout")
            )

        # batch norm params
        self.batchNorm = {"decay": self.configs.batch_norm_decay, "train": self.is_training}
        self.control_dim = params.model.control_dim
        self.memory_dim = params.model.memory_dim

    '''
    The Image Input Unit (stem). Passes the image features through a CNN-network
    Optionally adds position encoding (doesn't in the default behavior).
    Flatten the image into Height * Width "Knowledge base" array.

    Args:
        images: image input. [batchSize, height, width, inDim]
        inDim: input image dimension
        outDim: image out dimension
        addLoc: if not None, adds positional encoding to the image

    Returns preprocessed images. 
    [batchSize, height * width, outDim]
    '''

    def stem(self, images, inDim, outDim, addLoc=None):
        cfg = self.configs.stem
        with tf.variable_scope("stem"):
            if addLoc is None:
                addLoc = cfg.location_aware

            if cfg.linear:
                features = ops.linear(images, inDim, outDim)
            else:
                dims = [inDim] + ([cfg.dim] * (cfg.num_layers - 1)) + [outDim]

                if addLoc:
                    images, inDim = ops.addLocation(
                        images, inDim, cfg.locationDim,
                        h=self.configs.image.output_height,
                        w=self.configs.image.output_width,
                        locType=cfg.locationType)
                    dims[0] = inDim

                features = multi_layer_cnn(
                    images, dims,
                    batch_norm=self.configs.stem.batch_norm,
                    dropout=self.placeholders.dropout,
                    kernel_sizes=cfg.kernel_sizes or cfg.kernel_size,
                    strides=cfg.stride_sizes or 1)

                if cfg.stemGridRnn:
                    features = ops.multigridRNNLayer(features, H, W, outDim)

            # flatten the 2d images into a 1d KB
            features = tf.reshape(features, (tf.shape(images)[0], -1, outDim))

        return features

    def build_embeddings(self, question_word_ids):
        word_embeddings = self.dataset.word_embeddings
        with tf.variable_scope("embeddings"):
            if self.configs.embedding.shared:
                embeddingsVar = tf.get_variable(
                    "emb", initializer=tf.to_float(embInit["qa"]),
                    dtype=tf.float32, trainable=self.configs.embedding.trainable)
                question_embs = tf.concat([tf.zeros((1, self.configs.embedding.dim)), embeddingsVar], axis=0)
                questions = tf.nn.embedding_lookup(question_embs, question_word_ids)
                answer_embs = tf.nn.embedding_lookup(question_embs, word_embeddings["ansMap"])
                return questions, question_embs, answer_embs
            else:
                # answer_embedding_table = tf.get_variable(
                #     "answer_embedding_table",
                #     initializer=tf.cast(self.dataset.word_embeddings['a'], tf.float32),
                #     trainable=self.configs.embedding.trainable)
                question_embedding_table = tf.get_variable(
                    "question_embedding_table",
                    initializer=tf.cast(self.dataset.word_embeddings['q'], tf.float32),
                    trainable=self.configs.embedding.trainable)
                question_word_embeddings = tf.nn.embedding_lookup(question_embedding_table, question_word_ids)
                return question_word_embeddings, question_embedding_table, None  # answer_embedding_table

    '''
    The Question Input Unit embeds the questions to randomly-initialized word vectors,
    and runs a recurrent bidirectional encoder (RNN/LSTM etc.) that gives back
    vector representations for each question (the RNN final hidden state), and
    representations for each of the question words (the RNN outputs for each word). 

    The method uses bidirectional LSTM, by default.
    Optionally projects the outputs of the LSTM (with linear projection / 
    optionally with some activation).
    
    Args:
        questions: question word embeddings  
        [batchSize, questionLength, wordEmbDim]

        questionLengths: the question lengths.
        [batchSize]

        projWords: True to apply projection on RNN outputs.
        projQuestion: True to apply projection on final RNN state.
        projDim: projection dimension in case projection is applied.  

    Returns:
        Contextual Words: RNN outputs for the words.
        [batchSize, questionLength, ctrlDim]

        Vectorized Question: Final hidden state representing the whole question.
        [batchSize, ctrlDim]
    '''
    def build_encoder(self, questions, question_lengths, projWords=False,
                projQuestion=False, projDim=None):
        cfg = self.configs.encoder
        with tf.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if cfg.encVariationalDropout:
                varDp = {"stateDp": self.placeholders.dropout,
                         "inputDp": self.placeholders.dropout,
                         "inputSize": self.configs.embedding.dim}

            # rnns
            for i in range(cfg.num_layers):
                question_contextual_word_embeddings, question_embeddings = ops.RNNLayer(
                    questions, question_lengths,
                    cfg.dim, bi=cfg.bidirectional, cellType=cfg.type,
                    dropout=self.placeholders.dropout, varDp=varDp,
                    name="rnn%d" % i)

            # dropout for the question vector
            question_embeddings = tf.nn.dropout(question_embeddings, rate=self.placeholders.dropout)

            # projection of encoder outputs 
            if projWords:
                question_contextual_word_embeddings = ops.linear(question_contextual_word_embeddings, cfg.dim, projDim,
                                               name="projCW")
            if projQuestion:
                question_embeddings = ops.linear(
                    question_embeddings, cfg.dim, projDim,
                    act=cfg.encProjQAct, name="projQ")

        return question_contextual_word_embeddings, question_embeddings

    '''
    Runs the MAC recurrent network to perform the reasoning process.
    Initializes a MAC cell and runs netLength iterations.
    
    Currently it passes the question and knowledge base to the cell during
    its creating, such that it doesn't need to interact with it through 
    inputs / outputs while running. The recurrent computation happens 
    by working iteratively over the hidden (control, memory) states.  
    
    Args:
        images: flattened image features. Used as the "Knowledge Base".
        (Received by default model behavior from the Image Input Units).
        [batchSize, H * W, memDim]

        question_embeddings: vector questions representations.
        (Received by default model behavior from the Question Input Units
        as the final RNN state).
        [batchSize, ctrlDim]

        questionWords: question word embeddings.
        [batchSize, questionLength, ctrlDim]

        question_contextual_word_embeddings: question contextual words.
        (Received by default model behavior from the Question Input Units
        as the series of RNN output states).
        [batchSize, questionLength, ctrlDim]

        questionLengths: question lengths.
        [batchSize]

    Returns the final control state and memory state resulted from the network.
    ([batchSize, ctrlDim], [bathSize, memDim])
    '''
    def build_mac(
            self,
            question_embeddings,
            question_word_embeddings,
            question_contextual_word_embeddings,
            question_lengths,
            knowledge_base,
            name="",
            reuse=None):
        cfg = self.configs.mac
        batch_size = tf.shape(question_lengths)[0]
        with tf.variable_scope("mac" + name, reuse=reuse):
            # initialize knowledge base
            # optionally merge question into knowledge base representation
            if cfg.initKBwithQ is not None:
                iVecQuestions = ops.linear(question_embeddings, self.control_dim, self.memory_dim, name="questions")

                concatMul = (cfg.initKBwithQ == "MUL")
                cnct, dim = ops.concat(knowledge_base, iVecQuestions, self.memory_dim, mul=concatMul, expandY=True)
                knowledge_base = ops.linear(cnct, dim, self.memory_dim, name="initKB")

            # initialize memory variational dropout mask
            if self.configs.mac.memory_variational_dropout:
                self.memDpMask = ops.generateVarDpMask((batch_size, self.memory_dim), self.placeholders.dropout)

            mac_cell = MACCell(
                self.params,
                self.placeholders,
                question_embeddings,
                question_word_embeddings,
                question_contextual_word_embeddings,
                question_lengths,
                knowledge_base,
                reuse=reuse)

            state = mac_cell.zero_state(batch_size)

            for i in range(self.configs.max_step):
                mac_cell.iteration = i
                state = mac_cell(state)

            final_control = state.control
            final_memory = state.memory

        return final_control, final_memory

    def build_output(self, memory, question_embeddings, images):
        cfg = self.configs.classifier
        with tf.variable_scope("outputUnit"):
            features = memory
            dim = self.memory_dim

            if cfg.use_question:
                eVecQuestions = ops.linear(question_embeddings, self.control_dim, self.memory_dim, name="outQuestion")
                features, dim = ops.concat(features, eVecQuestions, self.memory_dim, mul=cfg.outQuestionMul)

            if cfg.use_image:
                images, imagesDim = ops.linearizeFeatures(
                    images,
                    self.configs.image.output_height,
                    self.configs.image.output_width,
                    self.configs.image.output_num_channels,
                    outDim=cfg.image_output_dim)
                images = ops.linear(images, self.memory_dim, cfg.image_output_dim, name="outImage")
                features = tf.concat([features, images], axis=-1)
                dim += cfg.outImageDim

        return features, dim

    '''
    Output Unit (step 2): Computes the logits for the answers. Passes the features
    through fully-connected network to get the logits over the possible answers.
    Optionally uses answer word embeddings in computing the logits (by default, it doesn't).

    Args:
        features: features used to compute logits
        [batchSize, inDim]

        inDim: features dimension

        aEmbedding: supported word embeddings for answer words in case answerMod is not NON.
        Optionally computes logits by computing dot-product with answer embeddings.
    
    Returns: the computed logits.
    [batchSize, answerWordsNum]
    '''
    def build_classifier(self, features, input_dim: int, aEmbeddings=None):
        cfg = self.configs.classifier
        with tf.variable_scope("classifier"):
            dims = [input_dim] + cfg.dims + [self.dataset.num_classes]
            if cfg.answerMod is not None:
                dims[-1] = cfg.wrdEmbDim

            logits = ops.FCLayer(
                features, dims,
                batchNorm=self.batchNorm if cfg.outputBN else None,
                dropout=self.placeholders.dropout)

            if cfg.answerMod is not None:
                logits = tf.nn.dropout(logits, rate=self.placeholders.dropout)
                interactions = ops.mul(aEmbeddings, logits, dims[-1], interMod=cfg.answerMod)
                logits = ops.inter2logits(interactions, dims[-1], sumMod="SUM")
                logits += ops.getBias((outputDim,), "ans")

        return logits

    def build(self, batch):
        cfg = self.configs
        images = tf.transpose(batch.images, (0, 2, 3, 1))
        # embed questions words (and optionally answer words)
        question_word_embeddings, qEmbeddings, aEmbeddings = self.build_embeddings(batch.questions)

        projWords = projQuestion = ((cfg.encoder.dim != self.control_dim) or cfg.encoder.project)
        question_contextual_word_embeddings, question_embeddings = self.build_encoder(
            question_word_embeddings,
            batch.question_lengths,
            projWords,
            projQuestion,
            self.control_dim)

        # Image Input Unit (stem)
        image_features = self.stem(images, self.configs.image.output_num_channels, self.memory_dim)

        final_control, final_memory = self.build_mac(
            question_embeddings, 
            question_word_embeddings,
            question_contextual_word_embeddings, 
            batch.question_lengths,
            image_features)

        output, dim = self.build_output(final_memory, question_embeddings, images)
        logits = self.build_classifier(output, dim, aEmbeddings)
        preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        return logits, preds

    def populate_feed_dict(self, feed_dict: Dict[tf.placeholder, Any], is_training):
        cfg = self.configs
        if is_training:
            feed_dict[self.placeholders.dropout] = cfg.dropout
        else:
            feed_dict[self.placeholders.dropout] = 0.0
        return super().populate_feed_dict(feed_dict, is_training)

    @property
    def references(self):
        return self.placeholders.answers

    def get_loss(self, batch, output):
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch.answers, logits=output)
            loss = tf.reduce_mean(losses)
        return loss

    def get_metric_ops(self):
        return dict(
            **super().get_metric_ops(),
            acc=tf.metrics.accuracy(self.references, self.predictions))

    @property
    def batch_size(self):
        return tf.shape(self.placeholders.question_lengths)[0]