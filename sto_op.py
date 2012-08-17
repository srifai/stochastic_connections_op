import theano
import theano.tensor as T
import numpy
from theano.tensor import as_tensor_variable,tensor,scal
from theano import Apply, Op
from theano import gof
from theano.gof import Apply, Constant, Op, Type, Variable

def sigmoid(x):
    return 1./(1.+numpy.exp(-x))

class StochasticConnectionsGrad(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, gz):
        assert isinstance(x, Variable)
        assert isinstance(gz, Variable)
        gx = tensor(dtype=scal.upcast(gz.dtype, x.dtype),
                    broadcastable=x.broadcastable)
        op = self
        return Apply(op, [x, gz], [gx])

    def perform(self, node, inp, out):
        x, gz = inp
        gx = out[0]
        b = numpy.array( [ [numpy.random.binomial(1,x[i,j]) for j in range(x.shape[1]) ] for i in range(x.shape[0]) ],dtype=numpy.bool)
        gx[0] = numpy.mean(numpy.array([ gz[i]*(b-x) for i in range(gz.shape[0]) ]),axis=0)
        assert gx[0].shape == x.shape

sc_grad = StochasticConnectionsGrad()

class StochasticConnectionsOp(theano.Op):
    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)

        if len(inputs) != 2:
            raise TypeError(
                "Wrong number of inputs for %s (got %i, expected 2)" %
                self)
        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by = i_broadcastables
        if len(bx) == 0:     # x is a scalar
            bz = by
        else:
            if len(by) >= 2:  # y is a matrix or tensor
                bz = bx[:-1] + by[:-2] + by[-1:]
            elif len(by) == 1:  # y is vector
                bz = bx[:-1]
            else:  # y is a scalar
                bz = bx

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor(scal.upcast(*i_dtypes), bz)]

        return theano.Apply(self, inputs, outputs)


    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    # Python implementation:
    def perform(self, node, inp, out):
        x, y = inp
        z, = out
        try:
            b = numpy.array( [ [numpy.random.binomial(1,sigmoid(y[i,j])) for j in range(y.shape[1]) ] for i in range(y.shape[0]) ],dtype=numpy.bool)
            z[0] = numpy.asarray(numpy.dot(x, numpy.sign(y)*b))
        except ValueError, e:
            if 1:
                raise ValueError('dot product failed.\n'
                                 'First arg dims: ' + str(x.shape) + '\n'
                                 'Second arg dims: ' + str(y.shape) + '\n'
                                 'First arg: \n' +
                                 min_informative_str(node.inputs[0]) +
                                 '\nSecond arg: \n' +
                                 min_informative_str(node.inputs[1]))
            e.args = e.args + (x.shape, y.shape)
            raise

    def grad(self, inp, grads):
        x, y = inp
        gz, = grads
        rval = sc(gz, y.T), sc_grad(T.nnet.sigmoid(y),gz)
        return T.cast(rval[0], x.dtype), T.cast(rval[1], y.dtype)

    # C implementation: [see theano web site for other functions]
    # def c_code(self,*args):
    #    pass

sc = StochasticConnectionsOp()

if __name__ == "__main__" :
    x = T.matrix()
    W = theano.shared(value = numpy.random.random(size=(5,10)).astype('float32'))
    h = sc(x,W)
    dhdW = T.grad(T.sum(h),x)
    f = theano.function([x],dhdW)
    print f(numpy.ones((2,5)).astype('float32'))
