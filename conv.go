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
	if inshape.Dims() != 4 || inshape.Dims() == 0 {
		return nil, fmt.Errorf("Expected shape is either a vector or a matrix, got %v", inshape)
	}

	l := &Conv{
		act:         gorgonia.Rectify,
		kernelShape: tensor.Shape{5, 5},
		pad:         []int{1, 1},
		stride:      []int{1, 1},
		dilation:    []int{1, 1},
	}

	for _, opt := range opts {
		var (
			o  Layer
			ok bool
		)

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

	l.w = gorgonia.NewTensor(g, of, 4, gorgonia.WithShape(l.size[0], l.size[1], l.kernelShape[0], l.kernelShape[1]), gorgonia.WithName("w"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	l.initialized = true

	return nil
}

// Conv represents a convolution layer
type Conv struct {
	w *gorgonia.Node

	name string
	size []int

	kernelShape           tensor.Shape
	pad, stride, dilation []int

	// optional config
	dropout *float64 // nil when shouldn't be applied

	act ActivationFunction

	initialized bool
}

// SetDropout sets the dropout of the layer
func (l *Conv) SetDropout(d float64) error {
	l.dropout = &d
	return nil
}

// SetSize sets the size of the layer
func (l *Conv) SetSize(s ...int) error {
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
		return wrapErr(l, "checking input: %w", err)
	}

	c, err := gorgonia.Conv2d(x.Node(), l.w, l.kernelShape, l.pad, l.stride, l.dilation)
	if err != nil {
		return wrapErr(l, "applying conv2d %v %v: %w", x.Node().Shape(), l.w.Shape(), err)
	}

	result, err := l.act(c)
	if err != nil {
		return wrapErr(l, "applying activation function: %w", err)
	}

	if l.dropout != nil {
		result, err = gorgonia.Dropout(result, *l.dropout)
		if err != nil {
			return wrapErr(l, "applying dropout: %w", err)
		}
	}

	logf("%T shape %s: %v", l, l.name, result.Shape())

	return result
}

// Type will return the hm.Type of the convolution layer
func (l *Conv) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'))
}

// Shape will return the tensor.Shape of the convolution layer
func (l *Conv) Shape() tensor.Shape {
	return l.w.Shape()
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
	_ namesetter      = &Conv{}
	_ actSetter       = &Conv{}
	_ dropoutConfiger = &Conv{}
	_ Term            = &Conv{}
)
