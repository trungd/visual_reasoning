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
    "images questions question_lengths answers "
    "dropout_read dropout_write dropout_encoder_input dropout_encoder_state dropout_memory dropout_question "
    "dropout_stem dropout_classifier")


class MACCell:
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
        self.reuse = reuse
        
        self.control_dim = params.model.control_dim
        self.memory_dim = params.model.memory_dim

        self.iteration = None
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}

        self._memory_dropout_mask = self.get_variational_dropout_mask(self.placeholders.dropout_memory)

    def get_variational_dropout_mask(self, rate):
        randomTensor = tf.to_float(1 - rate)
        randomTensor += tf.random_uniform((tf.shape(self.knowledge_base)[0], self.memory_dim), minval=0, maxval=1)
        binaryTensor = tf.floor(randomTensor)
        return tf.to_float(binaryTensor)

    def apply_variational_dropout_mask(self, inp, mask, rate):
        ret = (tf.div(inp, tf.to_float(1 - rate))) * mask
        return ret

    @property
    def state_size(self):
        return MACCellTuple(self.control_dim, self.memory_dim)

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
    def control(
            self,
            control_input,
            question_lengths,
            control,
            name="control",
            reuse=tf.AUTO_REUSE):
        cfg = self.configs.control
        dim = self.control_dim

        with tf.variable_scope(name, reuse=reuse):
            words = self.question_contextual_word_embeddings
            inWords = outWords = words
            if cfg.project_input_words or cfg.project_output_words:
                pWords = ops.linear(words, dim, dim, name="wordsProj")
                inWords = pWords if cfg.project_input_words else words
                outWords = pWords if cfg.project_output_words else words

            # Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = control_input
            if cfg.feed_previous:
                newContControl = control if cfg.controlFeedPrevAtt else contControl
                if cfg.controlFeedInputs:
                    newContControl = tf.concat([newContControl, control_input], axis=-1)
                    dim += self.control_dim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, dim, act=cfg.controlContAct, name="contControl")
                dim = self.control_dim

            # Step 2: compute attention distribution over words and sum them up accordingly.
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
            logits = ops.linear(interactions, dim, 1, dropout=0., name="logits")
            # self.interL = (interW, interb)

            attention = tf.nn.softmax(ops.expMask(logits, question_lengths))
            self.attentions["question"].append(attention)

            new_control = ops.att2Smry(attention, outWords)

        return new_control

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

    def read(self, knowledge_base, memory, control, name="read", reuse=tf.AUTO_REUSE):
        cfg = self.configs.read
        with tf.variable_scope(name, reuse=reuse):
            dim = self.memory_dim

            # memory dropout
            if self.configs.memory_variational_dropout:
                memory = self.apply_variational_dropout_mask(
                    memory, self._memory_dropout_mask, self.placeholders.dropout_memory)
            else:
                memory = tf.nn.dropout(memory, rate=self.placeholders.dropout_memory)

            # Step 1: knowledge base / memory interactions
            # parameters for knowledge base and memory projection
            proj = None
            if cfg.project_inputs:
                proj = {"dim": cfg.attention_dim, "shared": cfg.share_project, "dropout": self.placeholders.dropout_read}
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

            # Step 2: compute interactions with control
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

                # optional non-linearity
                interactions = get_activation_fn(cfg.activation_fn)(interactions)

            # Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            attention = ops.inter2att(interactions, dim, dropout=self.placeholders.dropout_read)

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
    def write(
            self,
            memory,
            info,
            control,
            controls=None,
            name="write",
            reuse=tf.AUTO_REUSE):
        cfg = self.configs.write
        with tf.variable_scope(name, reuse=reuse):
            # optionally project info
            if cfg.project_info:
                info = ops.linear(info, self.memory_dim, self.memory_dim, name="info")

            # optional info non-linearity
            info = get_activation_fn(cfg.info_activation_fn)(info)

            # compute self-attention vector based on previous controls and memories
            if cfg.self_attention:
                selfControl = control
                if cfg.writeSelfAttMod == "CONT":
                    selfControl = contControl
                # elif cfg.writeSelfAttMod == "POST":
                #     selfControl = postControl
                selfControl = ops.linear(selfControl, self.control_dim, self.control_dim, name="ctrlProj")
                interactions = controls * tf.expand_dims(selfControl, axis=1)
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
                new_memory, dim = ops.concat(new_memory, info, dim)

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

                logits = ops.linear(interactions, dim, 1, dropout=0., name="logits")
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

    def __call__(self, controls, memories, scope=None):
        cfg = self.configs
        scope = scope or type(self).__name__
        dim = self.control_dim

        with tf.variable_scope(scope, reuse=self.reuse):  # as tfscope
            control = controls[-1]
            memory = memories[-1]

            # control unit
            control_input = ops.linear(self.question_embeddings, dim, dim, reuse=self.iteration > 0)
            control_input = get_activation_fn(cfg.control.input_activation_fn)(control_input)
            if cfg.control.share_input_layer:
                control_input = ops.linear(control_input, dim, dim, reuse=self.iteration > 0)
            else:
                control_input = ops.linear(control_input, dim, dim, name=f"linear_{self.iteration}")

            # control
            new_control = self.control(control_input, self.question_lengths, control)

            # read
            info = self.read(self.knowledge_base, memory, new_control)

            # write
            info = tf.nn.dropout(info, rate=self.placeholders.dropout_write)
            new_memory = self.write(memory, info, new_control)
        return new_control, new_memory

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
    def init_state(self, name, dim, init_type, batch_size):
        if init_type == "prm":
            prm = tf.get_variable(name, shape=(dim,), initializer=tf.random_normal_initializer())
            init_state = tf.tile(tf.expand_dims(prm, axis=0), [batch_size, 1])
        elif init_type == "zero":
            init_state = tf.zeros((batch_size, dim), dtype=tf.float32)
        elif init_type == "questions":
            init_state = self.question_embeddings
        else:
            raise ValueError
        return init_state


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
                dropout_read=tf.placeholder(tf.float32, shape=(), name="dropout_read"),
                dropout_memory=tf.placeholder(tf.float32, shape=(), name="dropout_memory"),
                dropout_write=tf.placeholder(tf.float32, shape=(), name="dropout_write"),
                dropout_question=tf.placeholder(tf.float32, shape=(), name="dropout_question"),
                dropout_stem=tf.placeholder(tf.float32, shape=(), name="dropout_stem"),
                dropout_encoder_state=tf.placeholder(tf.float32, shape=(), name="dropout_encoder_state"),
                dropout_encoder_input=tf.placeholder(tf.float32, shape=(), name="dropout_encoder_input"),
                dropout_classifier=tf.placeholder(tf.float32, shape=(), name="dropout_classifier")
            )

        # batch norm params
        self.batch_norm = {"decay": self.configs.batch_norm_decay, "train": self.is_training}
        self.control_dim = params.model.control_dim
        self.memory_dim = params.model.memory_dim

    def populate_feed_dict(self, feed_dict: Dict[tf.placeholder, Any], is_training):
        cfg = self.configs
        ph = self.placeholders
        if is_training:
            feed_dict[ph.dropout_read] = cfg.mac.read.dropout or cfg.dropout
            feed_dict[ph.dropout_write] = cfg.mac.write.dropout or cfg.dropout
            feed_dict[ph.dropout_encoder_input] = cfg.encoder.dropout_input or cfg.dropout
            feed_dict[ph.dropout_encoder_state] = cfg.encoder.dropout_state or cfg.dropout
            feed_dict[ph.dropout_question] = cfg.encoder.dropout_question or cfg.dropout
            feed_dict[ph.dropout_stem] = cfg.stem.dropout or cfg.dropout
            feed_dict[ph.dropout_classifier] = cfg.classifier.dropout or cfg.dropout
            feed_dict[ph.dropout_memory] = cfg.mac.read.dropout_memory or cfg.dropout
        else:
            feed_dict[ph.dropout_read] = 0.
            feed_dict[ph.dropout_write] = 0.
            feed_dict[ph.dropout_encoder_input] = 0.
            feed_dict[ph.dropout_encoder_state] = 0.
            feed_dict[ph.dropout_question] = 0.
            feed_dict[ph.dropout_stem] = 0.
            feed_dict[ph.dropout_classifier] = 0.
            feed_dict[ph.dropout_memory] = 0.
        return super().populate_feed_dict(feed_dict, is_training)

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
                    dropout=self.placeholders.dropout_stem,
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

    def build_encoder(self, questions, question_lengths):
        cfg = self.configs.encoder
        dim = self.control_dim
        with tf.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if cfg.variational_dropout:
                varDp = {"stateDp": self.placeholders.dropout_encoder_state,
                         "inputDp": self.placeholders.dropout_encoder_input,
                         "inputSize": self.configs.embedding.dim}

            for i in range(cfg.num_layers):
                question_contextual_word_embeddings, question_embeddings = ops.rnn_layer(
                    questions, question_lengths,
                    cfg.dim, bidirectional=cfg.bidirectional, cell_type=cfg.type,
                    dropout=self.placeholders.dropout_encoder_input,
                    varDp=varDp,
                    name=f"{cfg.type.lower()}_{i}")

            # dropout for the question vector
            question_embeddings = tf.nn.dropout(question_embeddings, rate=self.placeholders.dropout_question)

            # projection of encoder outputs 
            if (cfg.dim != dim) or cfg.project:
                question_contextual_word_embeddings = ops.linear(
                    question_contextual_word_embeddings, cfg.dim, dim, name="projCW")
                question_embeddings = ops.linear(question_embeddings, cfg.dim, dim, act=cfg.encProjQAct, name="projQ")

        return question_contextual_word_embeddings, question_embeddings

    def build_mac(
            self,
            question_embeddings,
            question_word_embeddings,
            question_contextual_word_embeddings,
            question_lengths,
            knowledge_base):
        cfg = self.configs.mac
        batch_size = tf.shape(question_lengths)[0]
        with tf.variable_scope("mac"):
            if cfg.init_knowledge_base_with_question:
                iVecQuestions = ops.linear(question_embeddings, self.control_dim, self.memory_dim, name="questions")
                concatMul = (cfg.init_knowledge_base_with_question == "mul")
                cnct, dim = ops.concat(knowledge_base, iVecQuestions, self.memory_dim, mul=concatMul, expandY=True)
                knowledge_base = ops.linear(cnct, dim, self.memory_dim, name="initKB")

            mac_cell = MACCell(
                self.params,
                self.placeholders,
                question_embeddings,
                question_word_embeddings,
                question_contextual_word_embeddings,
                question_lengths,
                knowledge_base)

            initial_control = mac_cell.init_state("init_control", self.control_dim, cfg.init_control, batch_size)
            initial_memory = mac_cell.init_state("init_memory", self.memory_dim, cfg.init_memory, batch_size)

            controls = [initial_control]
            memories = [initial_memory]

            for i in range(self.configs.max_step):
                mac_cell.iteration = i
                new_control, new_memory = mac_cell(controls, memories)
                controls.append(new_control)
                memories.append(new_memory)

            final_control = controls[-1]
            final_memory = memories[-1]

        return final_control, final_memory

    def build_output(self, memory, question_embeddings, images):
        cfg = self.configs.classifier
        with tf.variable_scope("output"):
            features = memory
            dim = self.memory_dim

            if cfg.use_question:
                q_emb = ops.linear(question_embeddings, self.control_dim, dim, name="question_linear")
                features, dim = ops.concat(features, q_emb, dim)

            if cfg.use_image:
                images, imagesDim = ops.linearizeFeatures(
                    images,
                    self.configs.image.output_height,
                    self.configs.image.output_width,
                    self.configs.image.output_num_channels,
                    outDim=cfg.image_output_dim)
                images = ops.linear(images, dim, cfg.image_output_dim, name="outImage")
                features = tf.concat([features, images], axis=-1)
                dim += cfg.outImageDim

        return features, dim

    def build_classifier(self, features, input_dim: int, aEmbeddings=None):
        cfg = self.configs.classifier
        with tf.variable_scope("classifier"):
            dims = [input_dim] + cfg.dims + [self.dataset.num_classes]
            if cfg.answer_embedding:
                dims[-1] = self.configs.embedding.dim

            logits = ops.linear_layers(
                features, dims,
                batchNorm=self.batch_norm if cfg.batch_norm else None,
                dropout=self.placeholders.dropout_classifier,
                act="elu")

            if cfg.answer_embedding:
                logits = tf.nn.dropout(logits, rate=self.placeholders.dropout_classifier)
                interactions = ops.mul(aEmbeddings, logits, dims[-1], interMod=cfg.answer_embedding)
                # logits = ops.inter2logits(interactions, dims[-1], sumMod="SUM")
                logits += ops.getBias((outputDim,), "ans")

        return logits

    def build(self, batch):
        images = tf.transpose(batch.images, (0, 2, 3, 1))
        # embed questions words (and optionally answer words)
        question_word_embeddings, qEmbeddings, aEmbeddings = self.build_embeddings(batch.questions)

        question_contextual_word_embeddings, question_embeddings = self.build_encoder(
            question_word_embeddings,
            batch.question_lengths)

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

    @property
    def references(self):
        return self.placeholders.answers

    def get_loss(self, batch, output):
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch.answers, logits=output)
            loss = tf.reduce_mean(losses)
        return loss

    @property
    def batch_size(self):
        return tf.shape(self.placeholders.question_lengths)[0]

    def get_metrics(self):
        return dict(
            acc=tf.metrics.accuracy(self.references, self.predictions))