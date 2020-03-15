from dlex import Params
from dlex.datasets.tf import Dataset
from dlex.tf.models import BaseModelV1


class Baseline(BaseModelV1):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)

    '''
        Baseline approach:
        If baselineAtt is True, applies several layers (baselineAttNumLayers)
        of stacked attention to image and memory, when memory is initialized
        to the vector questions. See baselineAttLayer for further details.
        Otherwise, computes result output features based on image representation
        (baselineCNN), or question (baselineLSTM) or both.
        Args:
            questions: question vector representation
            [batch_size, questionDim]
            questionDim: dimension of question vectors
            images: (flattened) image representation
            [batch_size, imageDim]
            imageDim: dimension of image representations.

            hDim: hidden dimension to compute interactions between image and memory
            (for attention-based baseline).
        Returns final features to use in later classifier.
        [batch_size, outDim] (out dimension depends on baseline method)
        '''

    def baseline(self, questions, questionDim, images, imageDim, hDim):
        with tf.variable_scope("baseline"):
            if config.baselineAtt:
                memory = self.linear(questions, questionDim, hDim, name="qProj")
                images = self.linear(images, imageDim, hDim, name="iProj")

                for i in range(config.baselineAttNumLayers):
                    memory = self.baselineAttLayer(images, memory, hDim, hDim,
                                                   name="baseline%d" % i)
                memDim = hDim
            else:
                images, imagesDim = ops.linearizeFeatures(images, self.H, self.W,
                                                          imageDim, projDim=config.baselineProjDim)
                if config.baselineLSTM and config.baselineCNN:
                    memory = tf.concat([questions, images], axis=-1)
                    memDim = questionDim + imageDim
                elif config.baselineLSTM:
                    memory = questions
                    memDim = questionDim
                else:  # config.baselineCNN
                    memory = images
                    memDim = imageDim

        return memory, memDim

    '''
        Stacked Attention Layer for baseline. Computes interaction between images
        and the previous memory, and casts it back to compute attention over the 
        image, which in turn is summed up with the previous memory to result in the
        new one. 
        Args:
            images: input image.
            [batch_size, H * W, inDim]
            memory: previous memory value
            [batch_size, inDim]
            inDim: inputs dimension
            hDim: hidden dimension to compute interactions between image and memory
        Returns the new memory value.
        '''

    def baselineAttLayer(self, images, memory, inDim, hDim, name="", reuse=None):
        with tf.variable_scope("attLayer" + name, reuse=reuse):
            # projImages = ops.linear(images, inDim, hDim, name = "projImage")
            # projMemory = tf.expand_dims(ops.linear(memory, inDim, hDim, name = "projMemory"), axis = -2)
            # if config.saMultiplicative:
            #     interactions = projImages * projMemory
            # else:
            #     interactions = tf.tanh(projImages + projMemory)
            interactions, _ = ops.mul(images, memory, inDim, proj={"dim": hDim, "shared": False},
                                      interMod=config.baselineAttType)

            attention = ops.inter2att(interactions, hDim)
            summary = ops.att2Smry(attention, images)
            newMemory = memory + summary

        return newMemory

    def build(self):
        cfg = self.configs
        output, dim = self.baseline(
            questions, cfg.control_dim,
            self.images, self.imageInDim, cfg.attDim)