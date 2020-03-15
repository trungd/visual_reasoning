import collections
from typing import Any

import tensorflow.compat.v1 as tf
from dlex import Params, Dict
from dlex.datasets.tf import Dataset
from dlex.tf.models.base_v1 import BaseModelV1
from dlex.tf.utils.tf_v1 import multi_layer_cnn, get_activation_fn, get_variable

from . import ops

MACCellTuple = collections.namedtuple("MACCellTuple", ("control", "memory"))
Placeholders = collections.namedtuple(
    "Placeholders",
    "images questions question_lengths answers dropout is_training")


def add_null_word(words, lengths, dim):
    batch_size = tf.shape(lengths)[0]
    nullWord = tf.get_variable(
        "zeroWord",
        shape=(1, dim),
        initializer=tf.random_normal_initializer())
    nullWord = tf.tile(tf.expand_dims(nullWord, axis=0), [batch_size, 1, 1])
    words = tf.concat([nullWord, words], axis=1)
    lengths += 1
    return words, lengths


class MACCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    '''Initialize the MAC cell.
    (Note that in the current version the cell is stateful --
    updating its own internals when being called)

    Args:
        questions: the vector representation of the questions.
        [batch_size, control_dim]
        question_word_embeddings: the question words embeddings.
        [batch_size, questionLength, control_dim]
        question_contextual_word_embeddings: the encoder outputs -- the "contextual" question words.
        [batch_size, questionLength, control_dim]
        question_lengths: the length of each question.
        [batch_size]

        memory_dropout: dropout on the memory state (Tensor scalar).
        read_dropout: dropout inside the read unit (Tensor scalar).
        write_dropout: dropout on the new information that gets into the write unit (Tensor scalar).

        batch_size: batch size (Tensor scalar).
        train: train or test mod (Tensor boolean).
        reuse: reuse cell
        knowledge_base:
    '''

    def __init__(self, is_training, dropout, params):
        self.params = params
        self.configs = params.model.mac
        self.control_dim = params.model.control_dim
        self.memory_dim = params.model.memory_dim

        self.is_training = is_training
        self.dropout = dropout

    @property
    def state_size(self):
        return MACCellTuple(self.configs.control_dim, self.configs.memory_dim)

    @property
    def output_size(self):
        return 1

    # pass encoder hidden states to control?
    """
    The Control Unit: computes the new control state -- the reasoning operation,
    by summing up the word embeddings according to a computed attention distribution.

    The unit is recurrent: it receives the whole question and the previous control state,
    merge them together (resulting in the "continuous control"), and then uses that 
    to compute attentions over the question words. Finally, it combines the words 
    together according to the attention distribution to get the new control state. 

    Args:
        control_input: external inputs to control unit (the question vector).
        [batch_size, control_dim]
        inWords: the representation of the words used to compute the attention.
        [batch_size, questionLength, control_dim]
        outWords: the representation of the words that are summed up. 
                  (by default inWords == outWords)
        [batch_size, questionLength, control_dim]
        question_lengths: the length of each question.
        [batch_size]
        control: the previous control hidden state value.
        [batch_size, control_dim]
        contControl: optional corresponding continuous control state
        (before casting the attention over the words).
        [batch_size, control_dim]
    Returns:
        the new control state
        [batch_size, control_dim]
        the continuous (pre-attention) control
        [batch_size, control_dim]
    """
    def control(
            self,
            control_input,      # [batch_size, control_dim]
            inWords,            # [batch_size, max_len, control_dim]
            outWords,           # [batch_size, max_len, control_dim]
            question_lengths,   # [batch_size]
            control,            # [batch_size, control_dim]
            contControl=None,   # [batch_size, control_dim]
            name="",
            reuse=None):
        cfg = self.configs
        with tf.variable_scope("control" + name, reuse=reuse):
            dim = self.control_dim
            # Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = control_input
            if cfg.controlFeedPrev:
                newContControl = control if cfg.controlFeedPrevAtt else contControl
                if cfg.controlFeedInputs:
                    newContControl = tf.concat([newContControl, control_input], axis=-1)
                    dim += self.control_dim

                # merge inputs together
                newContControl = ops.linear(
                    newContControl, dim, self.control_dim,
                    act=cfg.control.activation_fn, name="contControl")
                dim = self.control_dim

            # Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis=1) * inWords

            # optionally concatenate words
            if cfg.control.concat_words:
                interactions = tf.concat([interactions, inWords], axis=-1)
                dim += self.control_dim

            # optional projection
            if cfg.control.project:
                interactions = ops.linear(
                    interactions, dim, self.control_dim,
                    act=cfg.control.project_activation_fn)
                dim = self.control_dim

            # compute attention distribution over words and summarize them accordingly
            logits = ops.inter2logits(interactions, dim)
            # self.interL = (interW, interb)

            attention = tf.nn.softmax(ops.expMask(logits, question_lengths))
            self.attentions["question"].append(attention)
            control_state = ops.att2Smry(attention, outWords)
        return control_state, newContControl

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
        [batch_size, control_dim]
    Returns the information extracted.
    [batch_size, memDim]
    '''
    def read(
            self,
            knowledge_base,
            memory,
            control,
            reuse=None):
        cfg = self.configs
        with tf.variable_scope("read", reuse=reuse):
            dim = self.memory_dim

            # memory dropout
            if cfg.memoryVariationalDropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropout)
            else:
                memory = tf.nn.dropout(memory, rate=self.dropout)

            # Step 1: knowledge base / memory interactions
            # parameters for knowledge base and memory projection
            proj = None
            if cfg.read.project_inputs:
                proj = {"dim": cfg.read.attention_dim, "shared": cfg.readProjShared, "dropout": self.dropout}
                dim = cfg.read.attention_dim
            # parameters for concatenating knowledge base elements
            concat = {"x": cfg.read.memory_concat_knowledge_base, "proj": cfg.read.memory_concat_projection}
            # compute interactions between knowledge base and memory
            interactions, interaction_dim = ops.mul(
                x=knowledge_base,
                y=memory,
                dim=self.memory_dim,
                proj=proj,
                concat=concat,
                interMod=cfg.read.attention_type,
                name="memInter")
            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if cfg.read.project_memory:
                interactions = ops.linear(
                    interactions,
                    interaction_dim,
                    dim,
                    activation_fn=cfg.read.activation_fn,
                    name="memKbProj")
            else:
                dim = interaction_dim

            # Step 2: compute interactions with control
            if cfg.read.control:
                # compute interactions with control
                if self.control_dim != dim:
                    control = ops.linear(control, self.control_dim, dim, name="ctrlProj")

                interactions, interDim = ops.mul(
                    interactions, control, dim, interMod=cfg.read.attention_type,
                    concat={"x": cfg.readCtrlConcatInter}, name="ctrlInter")

                # optionally concatenate knowledge base elements
                if cfg.read.control_concat_knowledge_base:
                    if cfg.readCtrlConcatProj:
                        addedInp, addedDim = projectedKB, cfg.attDim
                    else:
                        addedInp, addedDim = knowledge_base, self.memory_dim
                    interactions = tf.concat([interactions, addedInp], axis=-1)
                    dim += addedDim

                    # optional nonlinearity
                interactions = get_activation_fn(cfg.read.activation_fn)(interactions)

            # Step 3: sum attentions up over the knowledge base
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
        [batch_size, control_dim]
        contControl: optional corresponding continuous control state 
        (before casting the attention over the words).
        [batch_size, control_dim]
    Return the new memory 
    [batch_size, memDim]
    '''
    def write(self, memory, info, control, contControl=None, name="", reuse=None):
        cfg = self.configs
        with tf.variable_scope("write" + name, reuse=reuse):

            # optionally project info
            if cfg.writeInfoProj:
                info = ops.linear(info, self.memory_dim, self.memory_dim, name="info")

            # optional info non-linearity
            info = get_activation_fn(cfg.write.info_activation_fn)(info)

            # compute self-attention vector based on previous controls and memories
            if cfg.write.self_attention:
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
            if cfg.write.inputs == "info":
                newMemory = info
            elif cfg.write.inputs == "sum":
                new_memory += info
            elif cfg.write.inputs == "info_sum":
                newMemory, dim = ops.concat(new_memory, info, dim, mul=cfg.writeConcatMul)

            if cfg.write.self_attention:
                newMemory = tf.concat([newMemory, selfSmry], axis=-1)
                dim += self.memory_dim

            if cfg.write.merge_control:
                newMemory = tf.concat([newMemory, control], axis=-1)
                dim += self.memory_dim

                # project memory back to memory dimension
            if cfg.write.project_memory or (dim != self.memory_dim):
                newMemory = ops.linear(newMemory, dim, self.memory_dim, name="newMemory")

            # optional memory nonlinearity
            newMemory = get_activation_fn(cfg.write.activation_fn)(newMemory)

            # write unit gate
            if cfg.write.memory_gate:
                gateDim = self.memory_dim
                if cfg.writeGateShared:
                    gateDim = 1

                z = tf.sigmoid(ops.linear(control, self.control_dim, gateDim, name="gate", bias=cfg.writeGateBias))
                self.attentions["gate"].append(z)
                newMemory = newMemory * z + memory * (1 - z)

            # optional batch normalization
            if cfg.write.memory_batch_norm:
                new_memory = tf.contrib.layers.batch_norm(
                    newMemory,
                    decay=cfg.batch_norm_decay,
                    center=cfg.batch_norm_center,
                    scale=cfg.batch_norm_scale,
                    is_training=self.is_training,
                    updates_collections=None)

        return new_memory

    def memAutoEnc(self, newMemory, info, control, name="", reuse=None):
        with tf.variable_scope("memAutoEnc" + name, reuse=reuse):
            # inputs to auto encoder
            features = info if cfg.autoEncMemInputs == "INFO" else newMemory
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
        MACCellTuple([batch_size, control_dim],[batch_size, memDim])
    Returns the new state -- the new memory and control values.
    MACCellTuple([batch_size, control_dim],[batch_size, memDim])
    '''

    def __call__(
            self,
            question_embeddings,
            question_word_embeddings,
            question_lengths,
            knowledge_base,
            state,
            iteration,
            scope=None):
        cfg = self.configs
        batch_size = tf.shape(question_embeddings)[0]
        with tf.variable_scope(scope or type(self).__name__):
            control = state.control
            memory = state.memory

            # cell sharing
            inputName = "qInput"
            inputNameU = "qInputU"
            inputReuseU = inputReuse = (iteration > 0)
            if cfg.control.share_input_layer:
                inputNameU = "qInput%d" % iteration
                inputReuseU = None

            cellReuse = (iteration > 0)
            if cfg.unsharedCells:
                cellName = str(iteration)
                cellReuse = None

            # control unit
            # prepare question input to control
            control_input = ops.linear(
                question_embeddings, self.control_dim, self.control_dim,
                dropout=self.dropout, actDropout=self.dropout,
                name=inputName, reuse=inputReuse)

            # project words
            self.inWords = self.outWords = question_word_embeddings
            if cfg.control_in_words_proj or cfg.control_out_words_proj:
                pWords = ops.linear(question_word_embeddings, self.control_dim, self.control_dim, name="wordsProj")
                self.inWords = pWords if cfg.control_in_words_proj else question_word_embeddings
                self.outWords = pWords if cfg.control_out_words_proj else question_word_embeddings

            control_input = get_activation_fn(cfg.control.input_activation_fn)(control_input)
            control_input = ops.linear(control_input, self.control_dim, self.control_dim, name=inputNameU)

            new_control, self.contControl = self.control(
                control_input,
                self.inWords,
                self.outWords,
                question_lengths,
                control,
                self.contControl)

            # read unit
            info = self.read(knowledge_base, memory, new_control, reuse=cellReuse)
            info = tf.nn.dropout(info, rate=self.dropout)
            new_memory = self.write(memory, info, new_control, self.contControl, reuse=cellReuse)

            # add auto encoder loss for memory
            # if cfg.autoEncMem:
            #     self.autoEncLosses["memory"] += memAutoEnc(newMemory, info, newControl)

            # append as standard list?
            self.controls = tf.concat([self.controls, tf.expand_dims(new_control, axis=1)], axis=1)
            self.memories = tf.concat([self.memories, tf.expand_dims(new_memory, axis=1)], axis=1)
            self.infos = tf.concat([self.infos, tf.expand_dims(info, axis=1)], axis=1)

            # self.contControls = tf.concat([self.contControls, tf.expand_dims(contControl, axis = 1)], axis = 1)
            # self.postControls = tf.concat([self.controls, tf.expand_dims(postControls, axis = 1)], axis = 1)

        new_state = MACCellTuple(new_control, new_memory)
        return tf.zeros((batch_size, 1), dtype=tf.float32), new_state

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
    def init_state(self, name, dim, init_type: str, batch_size):
        init_type = init_type.lower()
        if init_type == "prm":
            prm = tf.get_variable(
                name, shape=(dim,),
                initializer=tf.random_normal_initializer())
            return tf.tile(tf.expand_dims(prm, axis=0), [batch_size, 1])
        elif init_type == "zero":
            return tf.zeros((batch_size, dim), dtype=tf.float32)

    def zero_state(
            self,
            batch_size,
            question_embeddings):
        """
        Initializes the cell internal state (currently it's stateful). In particular,
            1. Data-structures (lists of attention maps and accumulated losses).
            2. The memory and control states.
            3. The knowledge base (optionally merging it with the question vectors)
            4. The question words used by the cell (either the original word embeddings, or the
               encoder outputs, with optional projection).
        :param batch_size:
        :return:
        """
        cfg = self.configs

        # initialize data-structures
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}

        # initialize state
        if self.configs.control_init == "questions":
            initialControl = question_embeddings
        else:
            initialControl = self.init_state("init_ctrl", self.control_dim, self.configs.control_init, batch_size)
        initialMemory = self.init_state("init_mem", self.memory_dim, self.configs.memory_init, batch_size)

        self.controls = tf.expand_dims(initialControl, axis=1)
        self.memories = tf.expand_dims(initialMemory, axis=1)
        self.infos = tf.expand_dims(initialMemory, axis=1)

        self.contControl = initialControl
        # self.contControls = tf.expand_dims(initialControl, axis = 1)
        # self.postControls = tf.expand_dims(initialControl, axis = 1)

        # if cfg.controlCoverage:
        #     self.coverage = tf.zeros((batch_size, tf.shape(words)[1]), dtype = tf.float32)
        #     self.coverageBias = tf.get_variable("coverageBias", shape = (),
        #         initializer = cfg.controlCoverageBias)

        # initialize memory variational dropout mask
        if cfg.memory_variational_dropout:
            self.memDpMask = ops.generateVarDpMask((batch_size, self.memory_dim), self.dropout)

        return MACCellTuple(initialControl, initialMemory)


class MAC(BaseModelV1):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)
        self._placeholders = Placeholders(
            images=tf.placeholder(tf.float32, shape=(
                None,
                params.model.image.output_height,
                params.model.image.output_width,
                params.model.image.output_num_channels)),
            questions=tf.placeholder(tf.int32, shape=(None, None)),
            question_lengths=tf.placeholder(tf.int32, shape=(None,)),
            answers=tf.placeholder(tf.int32, shape=(None,)),
            is_training=tf.placeholder(tf.bool, name="is_training"),
            dropout=tf.placeholder(tf.float32, shape=(), name="dropout")
        )

    '''
    The Image Input Unit (stem). Passes the image features through a CNN-network
    Optionally adds position encoding (doesn't in the default behavior).
    Flatten the image into Height * Width "Knowledge base" array.
    Args:
        images: image input. [batch_size, height, width, inDim]
        inDim: input image dimension
        outDim: image out dimension
        addLoc: if not None, adds positional encoding to the image
    Returns preprocessed images. 
    [batch_size, height * width, outDim]
    '''
    def stem(self, images, inDim, outDim, addLoc=None):
        cfg = self.configs.stem
        with tf.variable_scope("stem"):
            addLoc = addLoc or self.configs.location_aware

            if self.configs.stem.linear:
                features = ops.linear(images, inDim, outDim, dropout=self.placeholders.dropout)
            else:
                dims = [inDim] + ([self.configs.stem.dim] * (self.configs.stem.num_layers - 1)) + [outDim]

                if addLoc:
                    images, inDim = ops.addLocation(
                        images, inDim, self.configs.locationDim,
                        h=self.H, w=self.W, locType=self.configs.locationType)
                    dims[0] = inDim

                features = multi_layer_cnn(
                    images, dims,
                    batch_norm=self.configs.stem.batch_norm,
                    dropout=self.configs.dropout,
                    kernel_sizes=cfg.kernel_sizes or cfg.kernel_size,
                    strides=cfg.stride_sizes or 1)

                if self.configs.stem.grid_rnn:
                    features = ops.multigridRNNLayer(features, H, W, outDim)

            # flatten the 2d images into a 1d KB
            features = tf.reshape(features, (self.batch_size, -1, outDim))

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
        [batch_size, questionLength, wordEmbDim]
        question_lengths: the question lengths.
        [batch_size]
        projWords: True to apply projection on RNN outputs.
        projQuestion: True to apply projection on final RNN state.
        projDim: projection dimension in case projection is applied.  
    Returns:
        Contextual Words: RNN outputs for the words.
        [batch_size, questionLength, control_dim]
        Vectorized Question: Final hidden state representing the whole question.
        [batch_size, control_dim]
    '''
    def build_encoder(self, questions, question_lengths):
        cfg = self.configs
        with tf.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if self.configs.encVariationalDropout:
                varDp = {"stateDp": self.placeholders.dropout,
                         "inputDp": self.placeholders.dropout,
                         "inputSize": cfg.wrdEmbDim}

            # rnns
            for i in range(self.configs.encoder.num_layers):
                question_contextual_word_embeddings, questions = ops.RNNLayer(
                    questions, question_lengths,
                    self.configs.encoder.dim,
                    bi=self.configs.encoder.bidirectional,
                    cellType=self.configs.encoder.type,
                    dropout=self.placeholders.dropout,
                    varDp=varDp,
                    name="rnn_%d" % i)

            # dropout for the question vector
            questions = tf.nn.dropout(questions, rate=self.placeholders.dropout)

            # projection of encoder outputs
            cfg = self.configs
            if (cfg.encoder.dim != cfg.control_dim) or cfg.encoder.project:
                question_contextual_word_embeddings = ops.linear(
                    question_contextual_word_embeddings,
                    cfg.encoder.dim, cfg.control_dim, name="projCW")
                questions = ops.linear(
                    questions,
                    cfg.encoder.dim, cfg.control_dim, name="projQ")

        return question_contextual_word_embeddings, questions

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
        [batch_size, H * W, memDim]
        questions: vector questions representations.
        (Received by default model behavior from the Question Input Units
        as the final RNN state).
        [batch_size, control_dim]
        question_word_embeddings: question word embeddings.
        [batch_size, questionLength, control_dim]
        question_contextual_word_embeddings: question contextual words.
        (Received by default model behavior from the Question Input Units
        as the series of RNN output states).
        [batch_size, questionLength, control_dim]
        question_lengths: question lengths.
        [batch_size]
    Returns the final control state and memory state resulted from the network.
    ([batch_size, control_dim], [bathSize, memory_dim])
    '''
    def build_mac(
            self,
            question_embeddings,                    # [batch_size, control_dim]
            question_word_embeddings,               # [batch_size, max_len, control_dim]
            question_contextual_word_embeddings,    # [batch_size, max_len, control_dim]
            question_lengths,                       # [batch_size]
            knowledge_base,                         # [batch_size, H * W, memory_dim]
    ):
        cfg = self.configs
        mac_cell = MACCell(
            is_training=self.placeholders.is_training,
            dropout=self.placeholders.dropout,
            params=self.params)

        state = mac_cell.zero_state(
            self.batch_size,
            question_embeddings)

        if cfg.use_original_embeddings:
            question_contextual_word_embeddings = question_word_embeddings

        # optionally add parametric "null" word in the to all questions
        if cfg.add_null_word:
            question_word_embeddings, question_lengths = add_null_word(
                question_word_embeddings, question_lengths, cfg.control_dim)

        # initialize knowledge base
        # optionally merge question into knowledge base representation
        if cfg.knowledge_base_init is not None:
            iVecQuestions = ops.linear(question_embeddings, cfg.control_dim, cfg.memory_dim, name="questions")
            concatMul = (cfg.knowledge_base_init == "mul")
            cnct, dim = ops.concat(knowledge_base, iVecQuestions, cfg.memory_dim, mul=concatMul, expandY=True)
            knowledge_base = ops.linear(cnct, dim, cfg.memory_dim, name="initKB")

        for i in range(cfg.max_step):
            _, state = mac_cell(
                question_embeddings,
                question_contextual_word_embeddings,
                question_lengths,
                knowledge_base,
                state,
                iteration=i)

        return state.control, state.memory

    '''
    Output Unit (step 2): Computes the logits for the answers. Passes the features
    through fully-connected network to get the logits over the possible answers.
    Optionally uses answer word embeddings in computing the logits (by default, it doesn't).
    Args:
        features: features used to compute logits
        [batch_size, inDim]
        inDim: features dimension
        aEmbedding: supported word embeddings for answer words in case answerMod is not NON.
        Optionally computes logits by computing dot-product with answer embeddings.

    Returns: the computed logits.
    [batch_size, answerWordsNum]
    '''
    def classifier(self, features, inDim, aEmbeddings=None):
        cfg = self.configs
        with tf.variable_scope("classifier"):
            output_dim = len(self.dataset.answers)
            dims = [inDim] + cfg.classifier.dims + [output_dim]
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
                logits += get_variable("biases", (output_dim,), "ans")

        return logits

    def get_loss(self, batch, output):
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch.answers, logits=output)
            loss = tf.reduce_mean(losses)
        return loss

    def get_metric_ops(self):
        return dict(
            **super().get_metric_ops(),
            acc=tf.metrics.accuracy(self.references, self.predictions))

    def build(self, batch: Placeholders):
        cfg = self.configs

        question_word_embeddings, question_embedding_table, answer_embedding_table = \
            self.build_embeddings(batch.questions)
        question_contextual_word_embeddings, question_embeddings = self.build_encoder(
            question_word_embeddings,
            batch.question_lengths)
        images = self.stem(batch.images, cfg.image.output_num_channels, cfg.memory_dim)

        final_control, final_memory = self.build_mac(
            question_embeddings,
            question_word_embeddings,
            question_contextual_word_embeddings,
            batch.question_lengths,
            images)

        # Output Unit - step 1 (preparing classifier inputs)
        with tf.variable_scope("output_unit"):
            x = final_memory
            dim = cfg.memory_dim

            if cfg.classifier.use_question:
                eVecQuestions = ops.linear(question_embeddings, cfg.control_dim, cfg.memory_dim, name="outQuestion")
                x = tf.concat([x, eVecQuestions], axis=-1)
                dim += cfg.control_dim

            if cfg.classifier.use_image:
                images, imagesDim = ops.linearizeFeatures(
                    images,
                    cfg.image.output_height,
                    cfg.image.output_width,
                    cfg.image.output_num_channels,
                    output_dim=cfg.classifier.image_output_dim)
                images = ops.linear(images, cfg.memory_dim, cfg.classifier.image_output_dim, name="outImage")

                x = tf.concat([x, images], axis=-1)
                dim += cfg.outImageDim

        # Output Unit - step 2 (classifier)
        logits = self.classifier(x, dim, answer_embedding_table)
        preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        return logits, preds

    @property
    def references(self):
        return self.placeholders.answers

    @property
    def batch_size(self):
        return tf.shape(self.placeholders.question_lengths)[0]

    def populate_feed_dict(self, feed_dict: Dict[tf.placeholder, Any], is_training):
        cfg = self.configs
        if is_training:
            feed_dict[self.placeholders.dropout] = cfg.dropout
            feed_dict[self.placeholders.is_training] = True
        else:
            feed_dict[self.placeholders.dropout] = 0.0
            feed_dict[self.placeholders.is_training] = False