from __future__ import division
import math
import tensorflow.compat.v1 as tf
from dlex.tf.utils.tf_v1 import relu, get_variable, get_activation_fn, get_rnn_cell

from .mi_gru_cell import MiGRUCell
from .mi_lstm_cell import MiLSTMCell

eps = 1e-20
inf = 1e30

######################################### basics #########################################

'''
Multiplies input inp of any depth by a 2d weight matrix.  
'''
# switch with conv 1?
def multiply(inp, W):
    inDim = tf.shape(W)[0]
    outDim = tf.shape(W)[1] 
    newDims = tf.concat([tf.shape(inp)[:-1], tf.fill((1,), outDim)], axis=0)
    
    inp = tf.reshape(inp, (-1, inDim))
    output = tf.matmul(inp, W)
    output = tf.reshape(output, newDims)

    return output

'''
Concatenates x and y. Support broadcasting. 
Optionally concatenate multiplication of x * y
'''
def concat(x, y, dim, mul = False, extendY=False):
    if extendY:
        y = tf.expand_dims(y, axis = -2)
        # broadcasting to have the same shape
        y = tf.zeros_like(x) + y

    if mul:
        out = tf.concat([x, y, x * y], axis = -1)
        dim *= 3
    else:
        out = tf.concat([x, y], axis = -1)
        dim *= 2
    
    return out, dim

'''
Adds L2 regularization for weight and kernel variables.
'''
# add l2 in the tf way
def L2RegularizationOp(l2 = None):
    if l2 is None:
        l2 = config.l2
    l2Loss = 0
    names = ["weight", "kernel"]
    for var in tf.trainable_variables():
        if any((name in var.name.lower()) for name in names):
            l2Loss += tf.nn.l2_loss(var)
    return l2 * l2Loss

######################################### attention #########################################

'''
Transform vectors to scalar logits.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return matching scalar for each interaction.
[batchSize, N]
'''
sumMod = ["LIN", "SUM"]
def inter2logits(interactions, dim, sumMod = "LIN", dropout=0.0, name = "", reuse = None):
    with tf.variable_scope("inter2logits" + name, reuse=reuse):
        if sumMod == "SUM":
            logits = tf.reduce_sum(interactions, axis = -1)
        else: # "LIN"
            logits = linear(interactions, dim, 1, dropout=dropout, name = "logits")
    return logits

'''
Transforms vectors to probability distribution. 
Calls inter2logits and then softmax over these.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return attention distribution over interactions.
[batchSize, N]
'''
def inter2att(interactions, dim, dropout=0.0, name = "", reuse = None):
    with tf.variable_scope("inter2att" + name, reuse = reuse): 
        logits = inter2logits(interactions, dim, dropout=dropout)
        attention = tf.nn.softmax(logits)    
    return attention

'''
Sums up features using attention distribution to get a weighted average over them. 
'''
def att2Smry(attention, features):
    return tf.reduce_sum(tf.expand_dims(attention, axis = -1) * features, axis = -2)

####################################### activations ########################################

# Sample from Gumbel(0, 1)
def sampleGumbel(shape): 
    U = tf.random_uniform(shape, minval = 0, maxval = 1)
    return -tf.log(-tf.log(U + eps) + eps)

# Draw a sample from the Gumbel-Softmax distribution
def gumbelSoftmaxSample(logits, temperature): 
    y = logits + sampleGumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def gumbelSoftmax(logits, temperature, train): # hard = False
    # Sample from the Gumbel-Softmax distribution and optionally discretize.
    # Args:
    #    logits: [batch_size, n_class] unnormalized log-probs
    #    temperature: non-negative scalar
    #    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    # Returns:
    #    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    #    If hard=True, then the returned sample will be one-hot, otherwise it will
    #    be a probabilitiy distribution that sums to 1 across classes

    y = gumbelSoftmaxSample(logits, temperature)

    # k = tf.shape(logits)[-1]
    # yHard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    yHard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims = True)), y.dtype)
    yNew = tf.stop_gradient(yHard - y) + y

    if config.gumbelSoftmaxBoth:
        return y
    if config.gumbelArgmaxBoth:
        return yNew
    ret = tf.cond(train, lambda: y, lambda: yNew)
    
    return ret 

def softmaxDiscrete(logits, temperature, train):
    if config.gumbelSoftmax:
        return gumbelSoftmax(logits, temperature = temperature, train = train)
    else:
        return tf.nn.softmax(logits)

###################################### sequence helpers ######################################

'''
Casts exponential mask over a sequence with sequence length.
Used to prepare logits before softmax.
'''
def expMask(seq, seqLength):
    maxLength = tf.shape(seq)[-1]
    mask = (1 - tf.cast(tf.sequence_mask(seqLength, maxLength), tf.float32)) * (-inf)
    masked = seq + mask
    return masked

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
'''
def seq2SeqLoss(logits, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.to_float(mask))
    return loss

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
    acc1: accuracy per symbol 
    acc2: accuracy per sequence
'''
def seq2seqAcc(preds, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    corrects = tf.logical_and(tf.equal(preds, targets), mask)
    numCorrects = tf.reduce_sum(tf.cast(corrects, tf.int32), axis=1)
    
    acc1 = tf.to_float(numCorrects) / (tf.to_float(lengths) + eps) # add small eps instead?
    acc1 = tf.reduce_mean(acc1)  
    
    acc2 = tf.to_float(tf.equal(numCorrects, lengths))
    acc2 = tf.reduce_mean(acc2)      

    return acc1, acc2

########################################### linear ###########################################

'''
linear transformation.

Args:
    inp: input to transform
    inDim: input dimension
    outDim: output dimension
    dropout: dropout over input
    batchNorm: if not None, applies batch normalization to inputs
    addBias: True to add bias
    bias: initial bias value
    act: if not None, activation to use after linear transformation
    actLayer: if True and act is not None, applies another linear transformation on top of previous
    actDropout: dropout to apply in the optional second linear transformation
    retVars: if True, return parameters (weight and bias) 

Returns linear transformation result.
'''
# batchNorm = {"decay": float, "train": Tensor}
# actLayer: if activation is not non, stack another linear layer
# maybe change naming scheme such that if name = "" than use it as default_name (-->unique?)
def linear(
        inp: tf.Tensor,
        inDim,
        outDim,
        dropout=0.0,
        batchNorm=None,
        activation_fn=None,
        actLayer=False,
        actDropout=0.0,
        name="",
        reuse=None):
    with tf.variable_scope("linear" + name, reuse=tf.AUTO_REUSE):
        linear = tf.layers.Dense(outDim, activation=get_activation_fn(activation_fn))
        # W = get_variable("weights", (inDim, outDim) if outDim > 1 else (inDim, ))
        # b = get_variable("biases", (outDim, ) if outDim > 1 else (), initializer="zero") + bias
        
        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(
                inp,
                decay=batchNorm["decay"],
                center=True,
                scale=True,
                is_training=batchNorm["train"],
                updates_collections=None)
            # tf.layers.batch_normalization, axis -1 ?

        inp = tf.nn.dropout(inp, rate=dropout)
        output = linear(inp)

        # good?
        if activation_fn is not None and actLayer:
            output = linear(
                output,
                outDim,
                outDim,
                dropout=actDropout,
                batchNorm=batchNorm,
                actLayer=False,
                name=name + "_2",
                reuse=reuse)

    return output

'''
Computes Multi-layer feed-forward network.

Args:
    features: input features
    dims: list with dimensions of network. 
          First dimension is of the inputs, final is of the outputs.
    batchNorm: if not None, applies batchNorm
    dropout: dropout value to apply for each layer
    act: activation to apply between layers.
    NON, TANH, SIGMOID, RELU, ELU
'''
# no activation after last layer
# batchNorm = {"decay": float, "train": Tensor}
def FCLayer(
        features,
        dims,
        batchNorm=None,
        dropout=0.0,
        activation_fn="relu"):
    layersNum = len(dims) - 1
    
    for i in range(layersNum):
        features = linear(
            features,
            dims[i],
            dims[i + 1],
            name="fc_%d" % i,
            batchNorm=batchNorm,
            dropout=dropout,
            reuse=tf.AUTO_REUSE)
        # not the last layer
        if i < layersNum - 1: 
            features = get_activation_fn(activation_fn)(features)
    
    return features   

###################################### cnns ######################################

'''
Computes convolution.

Args:
    inp: input features
    inDim: input dimension
    outDim: output dimension
    batchNorm: if not None, applies batchNorm on inputs
    dropout: dropout value to apply on inputs
    addBias: True to add bias
    kernelSize: kernel size
    stride: stride size
    act: activation to apply on outputs
    NON, TANH, SIGMOID, RELU, ELU
'''
# batchNorm = {"decay": float, "train": Tensor, "center": bool, "scale": bool}
# collections.namedtuple("batchNorm", ("decay", "train"))


'''
Computes Multi-layer convolutional network.

Args:
    features: input features
    dims: list with dimensions of network. 
          First dimension is of the inputs. Final is of the outputs.
    batchNorm: if not None, applies batchNorm
    dropout: dropout value to apply for each layer
    kernelSizes: list of kernel sizes for each layer. Default to config.stemKernelSize
    strides: list of strides for each layer. Default to 1.
    act: activation to apply between layers.
    NON, TANH, SIGMOID, RELU, ELU
'''
# batchNorm = {"decay": float, "train": Tensor, "center": bool, "scale": bool}
# activation after last layer


######################################## location ########################################

'''
Computes linear positional encoding for h x w grid. 
If outDim positive, casts positions to that dimension.
'''
# ignores dim
# h,w can be tensor scalars
def locationL(h, w, dim, outDim = -1, addBias = True):
    dim = 2
    grid = tf.stack(tf.meshgrid(tf.linspace(-config.locationBias, config.locationBias, w), 
                                tf.linspace(-config.locationBias, config.locationBias, h)), axis = -1)

    if outDim > 0:
        grid = linear(grid, dim, outDim, addBias = addBias, name = "locationL")
        dim = outDim

    return grid, dim

'''
Computes sin/cos positional encoding for h x w x (4*dim). 
If outDim positive, casts positions to that dimension.
Based on positional encoding presented in "Attention is all you need"
'''
# dim % 4 = 0
# h,w can be tensor scalars
def locationPE(h, w, dim, outDim = -1, addBias = True):    
    x = tf.expand_dims(tf.to_float(tf.linspace(-config.locationBias, config.locationBias, w)), axis = -1)
    y = tf.expand_dims(tf.to_float(tf.linspace(-config.locationBias, config.locationBias, h)), axis = -1)
    i = tf.expand_dims(tf.to_float(tf.range(dim)), axis = 0)

    peSinX = tf.sin(x / (tf.pow(10000.0, i / dim)))
    peCosX = tf.cos(x / (tf.pow(10000.0, i / dim)))
    peSinY = tf.sin(y / (tf.pow(10000.0, i / dim)))
    peCosY = tf.cos(y / (tf.pow(10000.0, i / dim)))

    peSinX = tf.tile(tf.expand_dims(peSinX, axis = 0), [h, 1, 1])
    peCosX = tf.tile(tf.expand_dims(peCosX, axis = 0), [h, 1, 1])
    peSinY = tf.tile(tf.expand_dims(peSinY, axis = 1), [1, w, 1])
    peCosY = tf.tile(tf.expand_dims(peCosY, axis = 1), [1, w, 1]) 

    grid = tf.concat([peSinX, peCosX, peSinY, peCosY], axis = -1)
    dim *= 4
    
    if outDim > 0:
        grid = linear(grid, dim, outDim, addBias = addBias, name = "locationPE")
        dim = outDim

    return grid, dim

locations = {
    "L": locationL,
    "PE": locationPE
}

'''
Adds positional encoding to features. May ease spatial reasoning.
(although not used in the default model). 

Args:
    features: features to add position encoding to.
    [batchSize, h, w, c]

    inDim: number of features' channels
    lDim: dimension for positional encodings
    outDim: if positive, cast enhanced features (with positions) to that dimension
    h: features' height
    w: features' width
    locType: L for linear encoding, PE for cos/sin based positional encoding
    mod: way to add positional encoding: concatenation (CNCT), addition (ADD), 
            multiplication (MUL), linear transformation (LIN).
'''
mods = ["CNCT", "ADD", "LIN", "MUL"]
# if outDim = -1, then will be set based on inDim, lDim
def addLocation(features, inDim, lDim, outDim = -1, h = None, w = None, 
    locType = "L", mod = "CNCT", name = "", reuse = None): # h,w not needed
    
    with tf.variable_scope("addLocation" + name, reuse = reuse):
        batchSize = tf.shape(features)[0]
        if h is None:
            h = tf.shape(features)[1]
        if w is None:
            w = tf.shape(features)[2]
        dim = inDim

        if mod == "LIN":
            if outDim < 0:
                outDim = dim

            grid, _ = locations[locType](h, w, lDim, outDim = outDim, addBias = False)
            features = linear(features, dim, outDim, name = "LIN")
            features += grid  
            return features, outDim

        if mod == "CNCT":
            grid, lDim = locations[locType](h, w, lDim)
            # grid = tf.zeros_like(features) + grid
            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid], axis = -1)
            dim += lDim

        elif mod == "ADD":
            grid, _ = locations[locType](h, w, lDim, outDim = dim)
            features += grid    
        
        elif mod == "MUL": # MUL
            grid, _ = locations[locType](h, w, lDim, outDim = dim)

            if outDim < 0:
                outDim = dim

            grid = tf.tile(tf.expand_dims(grid, axis = 0), [batchSize, 1, 1, 1])
            features = tf.concat([features, grid, features * grid], axis = -1)
            dim *= 3                

        if outDim > 0:
            features = linear(features, dim, outDim)
            dim = outDim 

    return features, dim

# config.locationAwareEnd
# H, W, _ = config.imageDims
# projDim = config.stemProjDim
# k = config.stemProjPooling
# projDim on inDim or on out
# inDim = tf.shape(features)[3]

'''
Linearize 2d image to linear vector.

Args:
    features: batch of 2d images. 
    [batchSize, h, w, inDim]

    h: image height

    w: image width

    inDim: number of channels

    projDim: if not None, project image to that dimension before linearization

    outDim: if not None, project image to that dimension after linearization

    loc: if not None, add positional encoding:
        locType: L for linear encoding, PE for cos/sin based positional encoding
        mod: way to add positional encoding: concatenation (CNCT), addition (ADD), 
            multiplication (MUL), linear transformation (LIN).
        pooling: number to pool image with before linearization.

Returns linearized image:
[batchSize, outDim] (or [batchSize, (h / pooling) * (w /pooling) * projDim] if outDim not supported) 
'''
# loc = {"locType": str, "mod": str}
def linearizeFeatures(
        features,
        h,
        w,
        inDim,
        projDim=None,
        output_dim=None,
        loc=None,
        pooling=2):
    if loc is not None:
        features = addLocation(
            features,
            inDim,
            lDim=inDim,
            outDim=inDim,
            locType=loc["locType"],
            mod=loc["mod"])

    if projDim is not None:
        features = linear(features, dim, projDim)
        features = relu(features)
        dim = projDim

    if pooling > 1:
        poolingDims = [1, pooling, pooling, 1]
        features = tf.nn.max_pool(
            features,
            ksize=poolingDims,
            strides=poolingDims,
            padding="SAME")
        h /= pooling
        w /= pooling
  
    dim = h * w * dim  
    features = tf.reshape(features, (-1, dim))
    
    if output_dim is not None:
        features = linear(features, dim, output_dim)
        dim = output_dim

    return features, dim

################################### multiplication ###################################
# specific dim / proj for x / y
'''
"Enhanced" hadamard product between x and y:
1. Supports optional projection of x, and y prior to multiplication.
2. Computes simple multiplication, or a parametrized one, using diagonal of complete matrix (bi-linear) 
3. Optionally concatenate x or y or their projection to the multiplication result.

Support broadcasting

Args:
    x: left-hand side argument
    [batchSize, dim]

    y: right-hand side argument
    [batchSize, dim]

    dim: input dimension of x and y
    
    dropout: dropout value to apply on x and y

    proj: if not None, project x and y:
        dim: projection dimension
        shared: use same projection for x and y
        dropout: dropout to apply to x and y if projected

    interMod: multiplication type:
        "MUL": x * y
        "DIAG": x * W * y for a learned diagonal parameter W
        "BL": x' W y for a learned matrix W

    concat: if not None, concatenate x or y or their projection. 
    
    mulBias: optional bias to stabilize multiplication (x * bias) (y * bias)

Returns the multiplication result
[batchSize, outDim] when outDim depends on the use of proj and cocnat arguments.
'''
# proj = {"dim": int, "shared": bool, "dropout": float} # "act": str, "actDropout": float
## interMod = ["direct", "scalarW", "bilinear"] # "additive"
# interMod = ["MUL", "DIAG", "BL", "ADD"]
# concat = {"x": bool, "y": bool, "proj": bool}
def mul(
        x,
        y,
        dim,
        dropout=0.0,
        proj=None,
        interMod="mul",
        concat=None,
        mulBias=0.,
        extendY=True,
        name="",
        reuse=None):
    
    with tf.variable_scope("mul" + name, reuse=reuse):
        origVals = {"x": x, "y": y, "dim": dim}

        x = tf.nn.dropout(x, rate=dropout)
        y = tf.nn.dropout(y, rate=dropout)
        # projection
        if proj is not None:
            x = tf.nn.dropout(x, rate=proj.get("dropout", 0.0))
            y = tf.nn.dropout(y, rate=proj.get("dropout", 0.0))

            if proj["shared"]:
                xName, xReuse = "proj", None
                yName, yReuse = "proj", True
            else:
                xName, xReuse = "projX", None
                yName, yReuse = "projY", None

            x = linear(x, dim, proj["dim"], name=xName, reuse=xReuse)
            y = linear(y, dim, proj["dim"], name=yName, reuse=yReuse)
            dim = proj["dim"]
            projVals = {"x": x, "y": y, "dim": dim}
            proj["x"], proj["y"] = x, y

        if extendY:
            y = tf.expand_dims(y, axis = -2)
            # broadcasting to have the same shape
            y = tf.zeros_like(x) + y

        # multiplication
        if interMod == "mul":
            output = (x + mulBias) * (y + mulBias)
        elif interMod == "diag":
            W = get_variable("weights", (dim, )) # change initialization?
            b = get_variable("biases", (dim, ), initializer="zeros")
            activations = x * W * y + b
        elif interMod == "bl":
            W = get_variable("weights", (dim, dim))
            b = get_variable("biases", (dim, ), initializer="zero")
            output = multiply(x, W) * y + b
        elif interMod == "add":
            output = tf.tanh(x + y)
        # concatenation
        if concat is not None:
            concatVals = projVals if concat.get("proj", False) else origVals
            if concat.get("x", False):
                output = tf.concat([output, concatVals["x"]], axis = -1)
                dim += concatVals["dim"]

            if concat.get("y", False):
                output = ops.concat(output, concatVals["y"], extendY = extendY)
                dim += concatVals["dim"]

    return output, dim

######################################## rnns ########################################



'''
Runs an forward RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize). 

Returns the outputs sequence and final RNN state.  
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def fwRNNLayer(
        sequences,
        seq_lengths,
        hidden_dim,
        cellType=None,
        dropout=0.0,
        varDp=None,
        name=None):
    with tf.variable_scope(name or "rnn") as scope:
        batchSize = tf.shape(sequences)[0]

        cell = get_rnn_cell(hidden_dim, cellType)

        if varDp is not None:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                state_keep_prob=1 - varDp["stateDp"],
                input_keep_prob=1 - varDp["inputDp"],
                variational_recurrent=True,
                input_size=varDp["inputSize"],
                dtype=tf.float32)
        else:
            sequences = tf.nn.dropout(sequences, rate=dropout)
        
        initial_state = cell.zero_state(batchSize, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell, sequences,
            sequence_length=seq_lengths,
            initial_state=initial_state,
            swap_memory=True,
            scope=scope)
            
        if isinstance(final_state, tf.nn.rnn_cell.LSTMStateTuple):
            final_state = final_state.h

    return outputs, final_state

'''
Runs an bidirectional RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def biRNNLayer(
        sequences,
        seq_lengths,
        hidden_dim,
        cell_type=None,
        dropout=0.0,
        varDp=None,
        name="",
        reuse=None): # proj = None,
    with tf.variable_scope("birnnLayer" + name, reuse=reuse):
        batchSize = tf.shape(sequences)[0]

        with tf.variable_scope("fw"):
            cellFw = get_rnn_cell(hidden_dim, cell_type)
        with tf.variable_scope("bw"):
            cellBw = get_rnn_cell(hidden_dim, cell_type)
        
        if varDp is not None:
            cellFw = tf.contrib.rnn.DropoutWrapper(
                cellFw,
                state_keep_prob=varDp["stateDp"],
                input_keep_prob=varDp["inputDp"],
                variational_recurrent=True,
                input_size = varDp["inputSize"],
                dtype=tf.float32)
            
            cellBw = tf.contrib.rnn.DropoutWrapper(
                cellBw,
                state_keep_prob=varDp["stateDp"],
                input_keep_prob=varDp["inputDp"],
                variational_recurrent=True,
                input_size=varDp["inputSize"],
                dtype=tf.float32)
        else:
            sequences = tf.nn.dropout(sequences, rate=dropout)

        initialStateFw = cellFw.zero_state(batchSize, tf.float32)
        initialStateBw = cellBw.zero_state(batchSize, tf.float32)

        (outSeqFw, outSeqBw), (lastStateFw, lastStateBw) = tf.nn.bidirectional_dynamic_rnn(
            cellFw, cellBw, sequences,
            sequence_length=seq_lengths,
            initial_state_fw=initialStateFw,
            initial_state_bw=initialStateBw,
            swap_memory=True)

        if isinstance(lastStateFw, tf.nn.rnn_cell.LSTMStateTuple):
            lastStateFw = lastStateFw.h
            lastStateBw = lastStateBw.h  

        outputs = tf.concat([outSeqFw, outSeqBw], axis=-1)
        last_state = tf.concat([lastStateFw, lastStateBw], axis=-1)

    return outputs, last_state

# int(hDim / 2) for biRNN?
'''
Runs an RNN layer by calling biRNN or fwRNN.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    bi: true to run bidirectional rnn.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
def RNNLayer(
        inSeq,
        seqL,
        hDim,
        bi=False,
        cellType=None,
        dropout=0.0,
        varDp=None,
        name=""):  # proj = None
    with tf.variable_scope("rnn_layer_" + name, reuse=tf.AUTO_REUSE):
        rnn = biRNNLayer if bi else fwRNNLayer
        if bi:
            hDim = int(hDim / 2)
        return rnn(
            inSeq,
            seqL,
            hDim,
            cell_type=cellType,
            dropout=dropout,
            varDp=varDp)

# tf counterpart?
# hDim = config.moduleDim
def multigridRNNLayer(featrues, h, w, dim, name = "", reuse = None):
    with tf.variable_scope("multigridRNNLayer" + name, reuse = reuse):
        featrues = linear(featrues, dim, dim / 2, name = "i")

        output0 = gridRNNLayer(featrues, h, w, dim, right = True, down = True, name = "rd")
        output1 = gridRNNLayer(featrues, h, w, dim, right = True, down = False, name = "r")
        output2 = gridRNNLayer(featrues, h, w, dim, right = False, down = True, name = "d")
        output3 = gridRNNLayer(featrues, h, w, dim, right = False, down = False, name = "NON")

        output = tf.concat([output0, output1, output2, output3], axis = -1)
        output = linear(output, 2 * dim, dim, name = "o")

    return outputs

# h,w should be constants
def gridRNNLayer(features, h, w, dim, right, down, name = "", reuse = None):
    with tf.variable_scope("gridRNNLayer" + name):
        batchSize = tf.shape(features)[0]

        cell = get_rnn_cell(dim, reuse=reuse, cell_type="rnn")
        
        initialState = cell.zero_state(batchSize, tf.float32)
        
        inputs = [tf.unstack(row, w, axis = 1) for row in tf.unstack(features, h, axis = 1)]
        states = [[None for _ in range(w)] for _ in range(h)]

        iAxis = range(h) if down else (range(h)[::-1])
        jAxis = range(w) if right else (range(w)[::-1])

        iPrev = -1 if down else 1
        jPrev = -1 if right else 1

        prevState = lambda i,j: states[i][j] if (i >= 0 and i < h and j >= 0 and j < w) else initialState
        
        for i in iAxis:
            for j in jAxis:
                prevs = tf.concat((prevState(i + iPrev, j), prevState(i, j + jPrev)), axis = -1)
                curr = inputs[i][j]
                _, states[i][j] = cell(prevs, curr)

        outputs = [tf.stack(row, axis = 1) for row in states]
        outputs = tf.stack(outputs, axis = 1)

    return outputs

# tf seq2seq?
# def projRNNLayer(inSeq, seqL, hDim, labels, labelsNum, labelsDim, labelsEmb, name = "", reuse = None):
#     with tf.variable_scope("projRNNLayer" + name):
#         batchSize = tf.shape(features)[0]

#         cell = createCell(hDim, reuse = reuse)

#         projCell = ProjWrapper(cell, labelsNum, labelsDim, labelsEmb, # config.wrdEmbDim
#             feedPrev = True, dropout = 1.0, config,
#             temperature = 1.0, sample = False, reuse)
        
#         initialState = projCell.zero_state(batchSize, tf.float32)
        
#         if config.soft:
#             inSeq = inSeq

#             # outputs, _ = tf.nn.static_rnn(projCell, inputs, 
#             #     sequence_length = seqL, 
#             #     initial_state = initialState)

#             inSeq = tf.unstack(inSeq, axis = 1)                        
#             state = initialState
#             logitsList = []
#             chosenList = []

#             for inp in inSeq:
#                 (logits, chosen), state = projCell(inp, state)
#                 logitsList.append(logits)
#                 chosenList.append(chosen)
#                 projCell.reuse = True

#             logitsOut = tf.stack(logitsList, axis = 1)
#             chosenOut = tf.stack(chosenList, axis = 1)
#             outputs = (logitsOut, chosenOut)
#         else:
#             labels = tf.to_float(labels)
#             labels = tf.concat([tf.zeros((batchSize, 1)), labels], axis = 1)[:, :-1] # ,newaxis
#             inSeq = tf.concat([inSeq, tf.expand_dims(labels, axis = -1)], axis = -1)

#             outputs, _ = tf.nn.dynamic_rnn(projCell, inSeq, 
#                 sequence_length = seqL, 
#                 initial_state = initialState,
#                 swap_memory = True)

#     return outputs #, labelsEmb

############################### variational dropout ###############################

'''
Generates a variational dropout mask for a given shape and a dropout 
probability value.
'''
def generateVarDpMask(shape, keepProb):
    randomTensor = tf.to_float(keepProb)
    randomTensor += tf.random_uniform(shape, minval = 0, maxval = 1)
    binaryTensor = tf.floor(randomTensor)
    mask = tf.to_float(binaryTensor)
    return mask

'''
Applies the a variational dropout over an input, given dropout mask 
and a dropout probability value. 
'''
def applyVarDpMask(inp, mask, keepProb):
    ret = (tf.div(inp, tf.to_float(keepProb))) * mask
    return ret   
