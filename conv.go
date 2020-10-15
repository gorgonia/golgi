// +build !cuda

package golgi

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ConsConv is a Conv construction function. It takes a gorgonia.Input that has a *gorgonia.Node.
func ConsConv(in gorgonia.Input, opts ...ConsOpt) (retVal Layer, err error) {
	x := in.Node()
	if x == nil {
		return nil, fmt.Errorf("ConsConv expects a *Node. Got input %v of  %T instead", in, in)
	}

	inshape := x.Shape()
	if inshape.Dims() > 2 || inshape.Dims() == 0 {
		return nil, fmt.Errorf("Expected shape is either a vector or a matrix")
	}

	l := &Conv{}

	for _, opt := range opts {
		var o Layer
		var ok bool

		if o, err = opt(l); err != nil {
			return nil, err
		}

		if l, ok = o.(*Conv); !ok {
			return nil, fmt.Errorf("Construction Option returned a non Conv. Got %T instead", o)
		}
	}

	// prep
	if err = l.Init(x); err != nil {
		return nil, err
	}

	return l, nil
}

// Init will initialize the fully connected layer
func (l *Conv) Init(xs ...*gorgonia.Node) (err error) {
	x := xs[0]
	g := x.Graph()
	of := x.Dtype()

	l.w = gorgonia.NewTensor(g, of, 4, gorgonia.WithShape(l.size, 3, l.kernelShape[0], l.kernelShape[1]), gorgonia.WithName("w"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	l.initialized = true

	return nil
}

type maxPoolConfig struct {
	kernelShape tensor.Shape
	pad, stride []int
}

// Conv represents a convolution layer
type Conv struct {
	w *gorgonia.Node

	name string
	size int

	kernelShape           tensor.Shape
	pad, stride, dilation []int

	// optional config
	dropout  *float64       // nil when shouldn't be applied
	mpConfig *maxPoolConfig // nil when shouldn't be applied

	act ActivationFunction

	initialized bool
}

// SetKernelShape sets the kernel shape of the layer
func (l *Conv) SetKernelShape(s tensor.Shape) error {
	l.kernelShape = s
	return nil
}

// SetPad sets the pad of the layer
func (l *Conv) SetPad(p []int) error {
	l.pad = p
	return nil
}

// SetStride sets the stride of the layer
func (l *Conv) SetStride(s []int) error {
	l.stride = s
	return nil
}

// SetDilation sets the dilation of the layer
func (l *Conv) SetDilation(s []int) error {
	l.dilation = s
	return nil
}

// SetDropout sets the dropout of the layer
func (l *Conv) SetDropout(d float64) error {
	l.dropout = &d
	return nil
}

// SetMaxPool sets a MaxPool layer for this layer in the form Conv->Activation->MaxPool(optional)->Dropout(optional)
func (l *Conv) SetMaxPool(kernelShape tensor.Shape, pad, stride []int) error {
	l.mpConfig = &maxPoolConfig{
		kernelShape: kernelShape,
		pad:         pad,
		stride:      stride,
	}

	return nil
}

// SetSize sets the size of the layer
func (l *Conv) SetSize(s int) error {
	l.size = s
	return nil
}

// SetName sets the name of the layer
func (l *Conv) SetName(n string) error {
	l.name = n
	return nil
}

// SetActivationFn sets the activation function of the layer
func (l *Conv) SetActivationFn(act ActivationFunction) error {
	l.act = act
	return nil
}

// Model will return the gorgonia.Nodes associated with this convolution layer
func (l *Conv) Model() gorgonia.Nodes {
	return gorgonia.Nodes{
		l.w,
	}
}

// Fwd runs the equation forwards
func (l *Conv) Fwd(x gorgonia.Input) gorgonia.Result {
	if err := gorgonia.CheckOne(x); err != nil {
		return gorgonia.Err(fmt.Errorf("Fwd of Conv %v: %w", l.name, err))
	}

	c, err := gorgonia.Conv2d(x.Node(), l.w, l.kernelShape, l.pad, l.stride, l.dilation)
	if err != nil {
		return gorgonia.Err(err)
	}

	result, err := l.act(c)
	if err != nil {
		return gorgonia.Err(err)
	}

	if l.mpConfig != nil {
		result, err = gorgonia.MaxPool2D(result, l.mpConfig.kernelShape, l.mpConfig.pad, l.mpConfig.stride)
		if err != nil {
			return gorgonia.Err(err)
		}
	}

	if l.dropout != nil {
		result, err = gorgonia.Dropout(result, *l.dropout)
		if err != nil {
			return gorgonia.Err(err)
		}
	}

	return result
}

// Type will return the hm.Type of the convolution layer
func (l *Conv) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Shape will return the tensor.Shape of the convolution layer
func (l *Conv) Shape() tensor.Shape {
	return l.w.Shape() // is this correct?
}

// Name will return the name of the convolution layer
func (l *Conv) Name() string {
	return l.name
}

// Describe will describe a convolution layer
func (l *Conv) Describe() {
	panic("not implemented")
}

var (
	_ sizeSetter        = &Conv{}
	_ namesetter        = &Conv{}
	_ actSetter         = &Conv{}
	_ dropoutConfiger   = &Conv{}
	_ maxpoolConfigurer = &Conv{}
)
